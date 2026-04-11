"""Volatility-focused feature engineering for V3.

Features are organized into 6 categories:
1. Volatility (~14 features) — PRIMARY: vol levels, vol-of-vol, vol term structure
2. Returns (~6 features) — multi-scale returns
3. Price action (~6 features) — range, position, drawdown
4. Volume (~4 features) — volume ratios
5. Cross-sectional (~6 features) — relative to market/peers
6. Flow (~4 features) — foreign/institutional (if available)

Total: ~40 features (before selection)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class VolFeatureEngineer:
    """Compute volatility-focused features for the V3 trading system."""

    # Feature column names by category
    VOL_FEATURES = [
        "vol_cc_5d", "vol_cc_10d", "vol_cc_20d", "vol_cc_60d",
        "vol_gk_5d", "vol_gk_10d", "vol_gk_20d",
        "vol_parkinson_5d", "vol_parkinson_10d", "vol_parkinson_20d",
        "vol_of_vol_10d", "vol_of_vol_20d",
        "vol_ratio_5_20", "vol_ratio_10_60",
    ]

    RETURN_FEATURES = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "log_return_1d", "log_return_5d",
    ]

    PRICE_FEATURES = [
        "price_position_20d", "price_position_60d",
        "intraday_range", "body_ratio",
        "drawdown_20d", "drawdown_60d",
    ]

    VOLUME_FEATURES = [
        "volume_ratio_5d", "volume_ratio_10d", "volume_ratio_20d",
        "volume_change_1d",
    ]

    CROSS_SECTIONAL_FEATURES = [
        "market_return_1d", "market_vol_20d",
        "relative_return_1d", "relative_return_5d",
        "vol_rank_20d", "return_rank_5d",
    ]

    # Flow features are computed in flow_collector.py
    FLOW_FEATURES = [
        "foreign_net_ratio", "inst_net_ratio",
        "flow_momentum_5d", "flow_divergence",
    ]

    @classmethod
    def all_feature_names(cls, include_flow: bool = False) -> list[str]:
        features = (
            cls.VOL_FEATURES
            + cls.RETURN_FEATURES
            + cls.PRICE_FEATURES
            + cls.VOLUME_FEATURES
            + cls.CROSS_SECTIONAL_FEATURES
        )
        if include_flow:
            features += cls.FLOW_FEATURES
        return features

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features. Expects OHLCV DataFrame grouped by ticker."""
        df = df.sort_values(["ticker", "date"]).copy()

        logger.info(f"Computing features for {df['ticker'].nunique()} tickers...")

        df = self._compute_volatility(df)
        df = self._compute_returns(df)
        df = self._compute_price_action(df)
        df = self._compute_volume(df)
        df = self._compute_cross_sectional(df)
        df = self._clip_outliers(df)

        n_features = sum(1 for c in df.columns if c in self.all_feature_names(True))
        logger.info(f"Features computed: {n_features} columns")
        return df

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features — the core of V3."""
        g = df.groupby("ticker")

        # Close-to-close volatility (annualized std of log returns)
        log_ret = g["close"].transform(lambda x: np.log(x / x.shift(1)))
        for w in [5, 10, 20, 60]:
            df[f"vol_cc_{w}d"] = log_ret.groupby(df["ticker"]).transform(
                lambda x: x.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(252)
            )

        # Garman-Klass volatility (uses OHLC — more efficient estimator)
        for w in [5, 10, 20]:
            df[f"vol_gk_{w}d"] = g.apply(
                lambda grp: self._garman_klass(grp, w), include_groups=False
            ).reset_index(level=0, drop=True)

        # Parkinson volatility (high-low range based)
        for w in [5, 10, 20]:
            df[f"vol_parkinson_{w}d"] = g.apply(
                lambda grp: self._parkinson(grp, w), include_groups=False
            ).reset_index(level=0, drop=True)

        # Vol-of-vol (meta-volatility: how much does vol itself fluctuate?)
        for w in [10, 20]:
            df[f"vol_of_vol_{w}d"] = df.groupby("ticker")["vol_cc_20d"].transform(
                lambda x: x.rolling(w, min_periods=5).std()
            )

        # Vol term structure (short/long ratio — expansion signal)
        df["vol_ratio_5_20"] = df["vol_cc_5d"] / df["vol_cc_20d"].replace(0, np.nan)
        df["vol_ratio_10_60"] = df["vol_cc_10d"] / df["vol_cc_60d"].replace(0, np.nan)

        return df

    @staticmethod
    def _garman_klass(grp: pd.DataFrame, window: int) -> pd.Series:
        """Garman-Klass volatility estimator (annualized)."""
        log_hl = np.log(grp["high"] / grp["low"]) ** 2
        log_co = np.log(grp["close"] / grp["open"]) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return (gk.rolling(window, min_periods=max(3, window // 2)).mean() * 252).apply(
            lambda x: np.sqrt(max(x, 0))
        )

    @staticmethod
    def _parkinson(grp: pd.DataFrame, window: int) -> pd.Series:
        """Parkinson volatility estimator (annualized)."""
        log_hl_sq = np.log(grp["high"] / grp["low"]) ** 2
        pk = log_hl_sq / (4 * np.log(2))
        return (pk.rolling(window, min_periods=max(3, window // 2)).mean() * 252).apply(
            lambda x: np.sqrt(max(x, 0))
        )

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-scale return features."""
        g = df.groupby("ticker")["close"]

        for w in [1, 5, 10, 20]:
            df[f"return_{w}d"] = g.transform(lambda x: x.pct_change(w))

        df["log_return_1d"] = g.transform(lambda x: np.log(x / x.shift(1)))
        df["log_return_5d"] = g.transform(lambda x: np.log(x / x.shift(5)))

        return df

    def _compute_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price position, range, and drawdown features."""
        g = df.groupby("ticker")

        # Price position within N-day range [0, 1]
        for w in [20, 60]:
            rolling_high = g["high"].transform(lambda x: x.rolling(w, min_periods=5).max())
            rolling_low = g["low"].transform(lambda x: x.rolling(w, min_periods=5).min())
            denom = (rolling_high - rolling_low).replace(0, np.nan)
            df[f"price_position_{w}d"] = (df["close"] - rolling_low) / denom

        # Intraday range
        df["intraday_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Body ratio (close-open relative to range)
        total_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = (df["close"] - df["open"]) / total_range

        # Drawdown from rolling peak
        for w in [20, 60]:
            rolling_peak = g["close"].transform(lambda x: x.rolling(w, min_periods=5).max())
            df[f"drawdown_{w}d"] = (df["close"] - rolling_peak) / rolling_peak.replace(0, np.nan)

        return df

    def _compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume ratio and change features."""
        g = df.groupby("ticker")["volume"]

        for w in [5, 10, 20]:
            rolling_avg = g.transform(lambda x: x.rolling(w, min_periods=3).mean())
            df[f"volume_ratio_{w}d"] = df["volume"] / rolling_avg.replace(0, np.nan)

        df["volume_change_1d"] = g.transform(lambda x: x.pct_change(1))

        return df

    def _compute_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional features (relative to market)."""
        # Market-wide return and vol (per date)
        market_stats = df.groupby("date").agg(
            market_return_1d=("return_1d", "mean"),
            market_vol_20d=("vol_cc_20d", "mean"),
        ).reset_index()

        df = df.merge(market_stats, on="date", how="left")

        # Relative returns
        df["relative_return_1d"] = df["return_1d"] - df["market_return_1d"]
        df["relative_return_5d"] = df["return_5d"] - df.groupby("date")["return_5d"].transform("mean")

        # Cross-sectional ranks (percentile within each date)
        df["vol_rank_20d"] = df.groupby("date")["vol_cc_20d"].rank(pct=True)
        df["return_rank_5d"] = df.groupby("date")["return_5d"].rank(pct=True)

        return df

    def _clip_outliers(self, df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
        """Two-stage outlier clipping: winsorize then sigma-clip."""
        feature_cols = self.all_feature_names(include_flow=True)
        feature_cols = [c for c in feature_cols if c in df.columns]

        for col in feature_cols:
            values = df[col].dropna()
            if len(values) < 10:
                continue

            # Stage 1: Winsorize at 0.1/99.9 percentile
            lo, hi = values.quantile([0.001, 0.999])
            df[col] = df[col].clip(lo, hi)

            # Stage 2: Sigma-clip
            mean, std = df[col].mean(), df[col].std()
            if std > 1e-8:
                df[col] = df[col].clip(mean - sigma * std, mean + sigma * std)

        return df

    def compute_vol_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        baseline_window: int = 20,
    ) -> pd.DataFrame:
        """Compute volatility expansion target.

        target = realized_vol(t+1:t+horizon) / realized_vol(t-baseline:t) - 1

        >0: vol expanding (opportunity)
        <0: vol contracting (avoid)
        """
        df = df.sort_values(["ticker", "date"]).copy()

        # Current vol (backward-looking)
        g = df.groupby("ticker")["close"]
        log_ret = g.transform(lambda x: np.log(x / x.shift(1)))

        current_vol = log_ret.groupby(df["ticker"]).transform(
            lambda x: x.rolling(baseline_window, min_periods=10).std() * np.sqrt(252)
        )

        # Future vol (forward-looking) — ONLY for training, NOT for inference
        future_vol = log_ret.groupby(df["ticker"]).transform(
            lambda x: x.shift(-1).rolling(horizon, min_periods=3).std() * np.sqrt(252)
        )

        df["vol_target"] = (future_vol / current_vol.replace(0, np.nan)) - 1

        # Clip extreme targets
        df["vol_target"] = df["vol_target"].clip(-2.0, 5.0)

        n_valid = df["vol_target"].notna().sum()
        logger.info(f"Vol target computed: {n_valid} valid samples "
                     f"(mean={df['vol_target'].mean():.4f}, std={df['vol_target'].std():.4f})")

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = "vol_target",
        max_features: int = 35,
        max_corr: float = 0.85,
        include_flow: bool = False,
    ) -> list[str]:
        """Select best features by correlation with target, removing redundancy.

        Args:
            df: DataFrame with features and target.
            target_col: Target column name.
            max_features: Maximum number of features to select.
            max_corr: Maximum inter-feature correlation (higher = drop one).
            include_flow: Include flow features in candidates.

        Returns:
            List of selected feature column names.
        """
        candidates = self.all_feature_names(include_flow=include_flow)
        candidates = [c for c in candidates if c in df.columns]

        # Compute absolute correlation with target
        valid = df[candidates + [target_col]].dropna()
        if len(valid) < 100:
            logger.warning("Too few valid samples for feature selection")
            return candidates[:max_features]

        target_corr = valid[candidates].corrwith(valid[target_col]).abs()
        target_corr = target_corr.sort_values(ascending=False)

        # Greedy selection: pick highest-corr features, skip redundant
        selected = []
        for feat in target_corr.index:
            if len(selected) >= max_features:
                break

            if not selected:
                selected.append(feat)
                continue

            # Check correlation with already-selected features
            redundant = False
            for sel in selected:
                corr = valid[feat].corr(valid[sel])
                if abs(corr) > max_corr:
                    redundant = True
                    break

            if not redundant:
                selected.append(feat)

        logger.info(f"Feature selection: {len(candidates)} → {len(selected)} features "
                     f"(max_corr={max_corr}, top IC: {target_corr.iloc[0]:.4f})")

        return selected
