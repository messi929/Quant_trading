"""Feature engineering: returns, volatility, volume patterns, and raw features."""

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Computes base features from OHLCV data for model consumption.

    Does NOT compute traditional technical indicators (RSI, MACD, etc.).
    Instead focuses on raw statistical features that the model can
    learn new indicators from.
    """

    # Feature windows for multi-scale analysis
    WINDOWS = [5, 10, 20, 60]

    def __init__(self):
        self.feature_cols: list[str] = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for each ticker.

        Args:
            df: Processed OHLCV DataFrame with columns:
                date, open, high, low, close, volume, ticker, market

        Returns:
            DataFrame with added feature columns
        """
        logger.info(f"Computing features for {df['ticker'].nunique()} tickers")
        df = df.sort_values(["ticker", "date"]).copy()

        # Group by ticker for per-stock computation
        groups = df.groupby("ticker", group_keys=False)

        df = groups.apply(self._compute_ticker_features)
        df = df.reset_index(drop=True)

        # Record feature columns (exclude metadata)
        meta_cols = {"date", "open", "high", "low", "close", "volume", "ticker", "market", "sector"}
        self.feature_cols = [c for c in df.columns if c not in meta_cols]

        # Drop rows with NaN from rolling computations
        n_before = len(df)
        df = df.dropna(subset=self.feature_cols)
        logger.info(
            f"Features computed: {len(self.feature_cols)} features, "
            f"{n_before - len(df)} rows dropped (warmup period)"
        )
        return df

    def _compute_ticker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single ticker."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]

        # --- Returns ---
        df["return_1d"] = close.pct_change()
        df["log_return_1d"] = np.log(close / close.shift(1))

        for w in self.WINDOWS:
            df[f"return_{w}d"] = close.pct_change(w)
            df[f"log_return_{w}d"] = np.log(close / close.shift(w))

        # --- Volatility (realized) ---
        for w in self.WINDOWS:
            df[f"volatility_{w}d"] = df["log_return_1d"].rolling(w).std()

        # --- Price ratios (relative position) ---
        for w in self.WINDOWS:
            rolling_high = high.rolling(w).max()
            rolling_low = low.rolling(w).min()
            price_range = rolling_high - rolling_low
            df[f"price_position_{w}d"] = np.where(
                price_range > 0,
                (close - rolling_low) / price_range,
                0.5,
            )

        # --- Volume features ---
        for w in self.WINDOWS:
            vol_ma = volume.rolling(w).mean()
            df[f"volume_ratio_{w}d"] = np.where(
                vol_ma > 0,
                volume / vol_ma,
                1.0,
            )

        df["volume_change"] = volume.pct_change().clip(-10, 10)

        # --- Intraday features ---
        price_range = high - low
        df["intraday_range"] = np.where(
            close > 0,
            price_range / close,
            0,
        )
        df["body_ratio"] = np.where(
            price_range > 0,
            (close - opn) / price_range,
            0,
        )
        df["upper_shadow"] = np.where(
            price_range > 0,
            (high - np.maximum(close, opn)) / price_range,
            0,
        )
        df["lower_shadow"] = np.where(
            price_range > 0,
            (np.minimum(close, opn) - low) / price_range,
            0,
        )

        # --- Gap ---
        df["gap"] = np.where(
            close.shift(1) > 0,
            (opn - close.shift(1)) / close.shift(1),
            0,
        )

        # --- Moving average distances ---
        for w in self.WINDOWS:
            ma = close.rolling(w).mean()
            df[f"ma_distance_{w}d"] = np.where(
                ma > 0,
                (close - ma) / ma,
                0,
            )

        # --- Return autocorrelation ---
        for w in [5, 20]:
            df[f"return_autocorr_{w}d"] = (
                df["return_1d"].rolling(w).apply(
                    lambda x: x.autocorr() if len(x) > 2 else 0,
                    raw=False,
                )
            )

        # --- Cross-sectional rank features (will be computed later in dataset) ---

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of computed feature column names."""
        return self.feature_cols.copy()
