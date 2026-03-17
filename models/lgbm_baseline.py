"""LightGBM cross-sectional baseline model.

Simple, interpretable benchmark that predicts cross-sectional
relative returns (rank) instead of absolute direction.
Serves as a sanity check — if the deep learning ensemble
can't beat this, simplify the architecture.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("lightgbm not installed — LGBMBaseline will be unavailable")


class LGBMBaseline:
    """Cross-sectional LightGBM model for sector return prediction.

    Key design choices:
    - Predicts cross-sectional RANK of forward returns (not absolute)
    - Uses only the latest snapshot features (no sequence modeling)
    - Trains on rolling windows to avoid look-ahead bias
    - Feature importance reveals which factors actually matter
    """

    DEFAULT_PARAMS = {
        "objective": "lambdarank",  # Learning-to-rank for cross-sectional
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        n_rounds: int = 300,
        early_stopping_rounds: int = 30,
        prediction_horizon: int = 5,
    ):
        if not HAS_LGBM:
            raise ImportError("lightgbm is required for LGBMBaseline")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.n_rounds = n_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.prediction_horizon = prediction_horizon
        self.model: Optional[lgb.Booster] = None
        self.feature_names: list[str] = []

    def prepare_cross_sectional_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """Prepare cross-sectional training data.

        For each date, compute the forward return rank across all tickers.
        This is the target for the learning-to-rank model.

        Args:
            df: DataFrame with features, date, ticker, close columns
            feature_cols: Feature column names

        Returns:
            (prepared_df, feature_cols) with added target columns
        """
        df = df.sort_values(["date", "ticker"]).copy()

        # Forward return (absolute)
        df["fwd_return"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-self.prediction_horizon) / x - 1
        )

        # Cross-sectional rank (0-1, higher = better relative performance)
        df["fwd_rank"] = df.groupby("date")["fwd_return"].rank(pct=True)

        # Drop rows without forward return
        df = df.dropna(subset=["fwd_return", "fwd_rank"])

        # Group size per date (needed for LambdaRank)
        df["group_size"] = df.groupby("date")["ticker"].transform("count")

        self.feature_names = feature_cols
        return df, feature_cols

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict:
        """Train the LightGBM model.

        Args:
            train_df: Training data (output of prepare_cross_sectional_data)
            val_df: Validation data
            feature_cols: Feature column names

        Returns:
            Training metrics dict
        """
        self.feature_names = feature_cols

        # For regression mode (simpler, often works just as well)
        # Switch to regression if lambdarank doesn't converge
        use_rank = self.params.get("objective") == "lambdarank"

        if use_rank:
            target_col = "fwd_rank"
            train_groups = train_df.groupby("date").size().values
            val_groups = val_df.groupby("date").size().values

            train_ds = lgb.Dataset(
                train_df[feature_cols],
                label=train_df[target_col],
                group=train_groups,
            )
            val_ds = lgb.Dataset(
                val_df[feature_cols],
                label=val_df[target_col],
                group=val_groups,
                reference=train_ds,
            )
        else:
            target_col = "fwd_return"
            train_ds = lgb.Dataset(
                train_df[feature_cols],
                label=train_df[target_col],
            )
            val_ds = lgb.Dataset(
                val_df[feature_cols],
                label=val_df[target_col],
                reference=train_ds,
            )

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds),
            lgb.log_evaluation(50),
        ]

        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=self.n_rounds,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        top_10 = importance.head(10)
        logger.info("Top 10 features by gain:")
        for _, row in top_10.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")

        return {
            "best_iteration": self.model.best_iteration,
            "feature_importance": importance,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate cross-sectional predictions.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of predicted scores (higher = better expected relative return)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(df[self.feature_names])

    def generate_sector_signals(
        self,
        df: pd.DataFrame,
        n_sectors: int,
        sector_col: str = "sector",
    ) -> np.ndarray:
        """Generate sector-level signals from ticker predictions.

        Args:
            df: DataFrame with features and sector column
            n_sectors: Number of sectors
            sector_col: Sector column name

        Returns:
            (n_sectors,) array of sector scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        df = df.copy()
        df["lgbm_score"] = self.predict(df)

        # Average score per sector
        sector_scores = df.groupby(sector_col)["lgbm_score"].mean()

        # Cross-sectional z-score
        mean_s = sector_scores.mean()
        std_s = sector_scores.std() + 1e-8
        z_scores = (sector_scores - mean_s) / std_s

        return z_scores.values

    def save(self, path: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"LightGBM model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"LightGBM model loaded from {path}")

    def backtest_cross_sectional(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        train_window: int = 252,
        retrain_frequency: int = 63,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Walk-forward cross-sectional backtest.

        Trains on rolling windows and evaluates long-short portfolio
        based on top-K vs bottom-K ranked predictions.

        Args:
            df: Full dataset with features
            feature_cols: Feature columns
            train_window: Training window in trading days
            retrain_frequency: Retrain every N days
            top_k: Number of top/bottom sectors for long/short

        Returns:
            DataFrame with date, long_return, short_return, ls_return columns
        """
        # Use regression for simpler walk-forward
        self.params["objective"] = "regression"
        self.params["metric"] = "rmse"

        dates = sorted(df["date"].unique())
        results = []
        last_train_idx = -retrain_frequency  # Force initial training

        for i in range(train_window, len(dates) - self.prediction_horizon):
            current_date = dates[i]

            # Retrain periodically
            if i - last_train_idx >= retrain_frequency:
                train_dates = dates[max(0, i - train_window):i]
                val_dates = dates[max(0, i - retrain_frequency):i]

                train_mask = df["date"].isin(train_dates)
                val_mask = df["date"].isin(val_dates)

                train_data = df[train_mask].copy()
                val_data = df[val_mask].copy()

                # Compute forward returns for training
                train_data["fwd_return"] = train_data.groupby("ticker")["close"].transform(
                    lambda x: x.shift(-self.prediction_horizon) / x - 1
                )
                val_data["fwd_return"] = val_data.groupby("ticker")["close"].transform(
                    lambda x: x.shift(-self.prediction_horizon) / x - 1
                )
                train_data = train_data.dropna(subset=["fwd_return"])
                val_data = val_data.dropna(subset=["fwd_return"])

                if len(train_data) < 100 or len(val_data) < 50:
                    continue

                train_ds = lgb.Dataset(train_data[feature_cols], label=train_data["fwd_return"])
                val_ds = lgb.Dataset(val_data[feature_cols], label=val_data["fwd_return"], reference=train_ds)

                self.model = lgb.train(
                    self.params, train_ds,
                    num_boost_round=self.n_rounds,
                    valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)],
                )
                last_train_idx = i
                self.feature_names = feature_cols

            if self.model is None:
                continue

            # Predict on current date
            current_data = df[df["date"] == current_date].copy()
            if len(current_data) < top_k * 2:
                continue

            scores = self.predict(current_data)
            current_data["score"] = scores

            # Actual forward return
            fwd_date = dates[min(i + self.prediction_horizon, len(dates) - 1)]
            fwd_data = df[df["date"] == fwd_date][["ticker", "close"]].rename(
                columns={"close": "fwd_close"}
            )
            current_data = current_data.merge(fwd_data, on="ticker", how="left")
            current_data["actual_return"] = current_data["fwd_close"] / current_data["close"] - 1
            current_data = current_data.dropna(subset=["actual_return"])

            if len(current_data) < top_k * 2:
                continue

            # Long top-K, short bottom-K
            sorted_df = current_data.sort_values("score", ascending=False)
            long_ret = sorted_df.head(top_k)["actual_return"].mean()
            short_ret = sorted_df.tail(top_k)["actual_return"].mean()

            results.append({
                "date": current_date,
                "long_return": long_ret,
                "short_return": short_ret,
                "ls_return": long_ret - short_ret,
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            cum_ls = (1 + result_df["ls_return"]).cumprod() - 1
            logger.info(
                f"LightGBM L/S backtest: {len(result_df)} periods, "
                f"cumulative L/S return: {cum_ls.iloc[-1]:.2%}, "
                f"mean daily L/S: {result_df['ls_return'].mean():.4%}"
            )
        return result_df
