"""Data processing: cleaning, normalization, and missing value handling."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler


class DataProcessor:
    """Cleans and normalizes market data for model consumption."""

    def __init__(
        self,
        min_history_days: int = 500,
        outlier_std: float = 5.0,
    ):
        self.min_history_days = min_history_days
        self.outlier_std = outlier_std
        self.scalers: dict[str, RobustScaler] = {}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full processing pipeline.

        Args:
            df: Raw OHLCV DataFrame with columns:
                date, Open, High, Low, Close, Volume, ticker, market

        Returns:
            Cleaned and processed DataFrame
        """
        logger.info(f"Processing {len(df)} rows, {df['ticker'].nunique()} tickers")
        initial_tickers = df["ticker"].nunique()

        df = self._standardize_columns(df)
        df = self._remove_delisted(df)
        df = self._filter_min_history(df)
        df = self._handle_missing(df)
        df = self._remove_outliers(df)
        df = self._sort(df)

        final_tickers = df["ticker"].nunique()
        logger.info(
            f"Processing complete: {initial_tickers} → {final_tickers} tickers, "
            f"{len(df)} rows"
        )
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("open", "high", "low", "close", "volume", "date", "ticker", "market"):
                col_map[c] = cl
        df = df.rename(columns=col_map)

        required = {"date", "open", "high", "low", "close", "volume", "ticker"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _remove_delisted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove tickers with zero/negative prices (likely delisted)."""
        mask = df.groupby("ticker")["close"].transform(
            lambda x: (x <= 0).sum() / len(x)
        )
        bad_tickers = df.loc[mask > 0.1, "ticker"].unique()
        if len(bad_tickers) > 0:
            logger.info(f"Removing {len(bad_tickers)} likely delisted tickers")
            df = df[~df["ticker"].isin(bad_tickers)]
        return df

    def _filter_min_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove tickers with insufficient history."""
        counts = df.groupby("ticker")["date"].count()
        valid = counts[counts >= self.min_history_days].index
        removed = len(counts) - len(valid)
        if removed > 0:
            logger.info(
                f"Removed {removed} tickers with < {self.min_history_days} days"
            )
        return df[df["ticker"].isin(valid)]

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward fill then backward fill."""
        n_missing_before = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()

        # Forward fill within each ticker, then backward fill
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df.groupby("ticker")[price_cols].transform(
            lambda x: x.ffill().bfill()
        )
        # Volume: fill with 0
        df["volume"] = df["volume"].fillna(0)

        # Drop any remaining NaN rows
        df = df.dropna(subset=price_cols)

        n_missing_after = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
        logger.info(f"Missing values: {n_missing_before} → {n_missing_after}")
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme price changes (likely data errors)."""
        df["_return"] = df.groupby("ticker")["close"].pct_change()
        mean_ret = df["_return"].mean()
        std_ret = df["_return"].std()
        threshold = self.outlier_std * std_ret

        outlier_mask = df["_return"].abs() > threshold
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.info(
                f"Clipping {n_outliers} outlier returns "
                f"(>{self.outlier_std}σ = {threshold:.4f})"
            )
            df.loc[outlier_mask, "_return"] = np.clip(
                df.loc[outlier_mask, "_return"],
                mean_ret - threshold,
                mean_ret + threshold,
            )
        df = df.drop(columns=["_return"])
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Apply RobustScaler normalization per market.

        Args:
            df: DataFrame with features
            feature_cols: Columns to normalize
            fit: Whether to fit scalers (True for train, False for test)

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        for market in df["market"].unique():
            mask = df["market"] == market
            key = f"market_{market}"

            if fit:
                scaler = RobustScaler()
                df.loc[mask, feature_cols] = scaler.fit_transform(
                    df.loc[mask, feature_cols]
                )
                self.scalers[key] = scaler
            else:
                if key not in self.scalers:
                    raise ValueError(f"No fitted scaler for {key}. Call with fit=True first.")
                df.loc[mask, feature_cols] = self.scalers[key].transform(
                    df.loc[mask, feature_cols]
                )

        return df
