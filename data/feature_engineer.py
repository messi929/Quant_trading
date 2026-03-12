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

        # Cross-sectional rank features (per-date rank across all tickers)
        rank_features = ["return_1d", "return_5d", "return_20d", "volume_ratio_5d", "volatility_20d"]
        for feat in rank_features:
            if feat in df.columns:
                df[f"{feat}_rank"] = df.groupby("date")[feat].rank(pct=True)

        # Cross-asset proxy / market-regime features
        df = self.add_market_regime_features(df)

        # Alternative data features (IV proxy, vol skew, illiquidity)
        try:
            from data.alternative_data import AlternativeDataFeatures
            alt_fe = AlternativeDataFeatures()
            df = alt_fe.compute_all(df)
        except ImportError:
            logger.debug("alternative_data module not available — skipping alt features")
        except Exception as e:
            logger.warning(f"Alternative feature computation failed: {e}")

        # Record feature columns (exclude metadata)
        meta_cols = {
            "date", "open", "high", "low", "close", "volume",
            "ticker", "market", "sector",
        }
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

        # --- Momentum quality (quality / consistency of trend) ---
        for w in [10, 20]:
            returns_w = df["log_return_1d"].rolling(w)
            # Ratio of positive days (momentum consistency)
            df[f"pos_day_ratio_{w}d"] = returns_w.apply(
                lambda x: (x > 0).sum() / max(len(x), 1), raw=True
            )
            # Trend strength (|mean| / std — signal-to-noise ratio)
            df[f"trend_strength_{w}d"] = returns_w.mean().abs() / (
                returns_w.std() + 1e-8
            )

        # --- Reversal / mean-reversion signals ---
        # Short-term reversal (contrarian 1-week)
        df["reversal_1w"] = -df["return_5d"]
        # Medium-term momentum minus short-term (captures continuation vs exhaustion)
        df["reversal_momentum_spread"] = df["return_20d"] - df["return_5d"]
        # Z-score of daily return relative to recent distribution
        rolling_std_20 = df["log_return_1d"].rolling(20).std()
        df["zscore_1d"] = df["log_return_1d"] / (rolling_std_20 + 1e-8)

        # --- Volume-price relationship ---
        # On-balance volume proxy: sign of daily return × relative volume
        df["obv_direction"] = np.sign(df["return_1d"]) * df["volume_ratio_5d"]
        # Volume-price divergence: large volume move without matching price move → divergence
        for w in [5, 20]:
            price_change = close.pct_change(w)
            vol_change = volume.pct_change(w)
            df[f"vol_price_divergence_{w}d"] = np.where(
                price_change.abs() > 1e-8,
                vol_change / (price_change.abs() + 1e-8),
                0,
            ).clip(-10, 10)

        # --- Drawdown-based features ---
        for w in [20, 60]:
            rolling_high = close.rolling(w).max()
            df[f"drawdown_{w}d"] = np.where(
                rolling_high > 0,
                (close - rolling_high) / rolling_high,
                0,
            )

        return df

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-level regime features (cross-sectional).

        Computes market-wide breadth, return, and volatility from all tickers
        already present in *df*.  These serve as lightweight proxies for
        cross-asset / macro signals without requiring external data sources.

        Args:
            df: DataFrame that already contains per-ticker features, including
                at least the columns ``date``, ``return_1d``, ``log_return_1d``,
                ``return_5d``, and ``ticker``.

        Returns:
            df with additional market-regime columns merged in-place.
        """
        # Equal-weight market return for each date
        market_ret = (
            df.groupby("date")["return_1d"]
            .mean()
            .rename("market_return_1d")
        )
        # Cross-sectional return dispersion (daily vol across tickers)
        market_vol = (
            df.groupby("date")["log_return_1d"]
            .std()
            .rename("market_vol_1d")
        )
        # Breadth: fraction of stocks with a positive daily return
        breadth = (
            df.groupby("date")["return_1d"]
            .apply(lambda x: (x > 0).sum() / max(len(x), 1))
            .rename("market_breadth")
        )

        # Merge back to per-row level
        df = df.merge(market_ret.reset_index(), on="date", how="left")
        df = df.merge(market_vol.reset_index(), on="date", how="left")
        df = df.merge(breadth.reset_index(), on="date", how="left")

        # Rolling market momentum (smooth the daily market return per ticker timeline)
        for w in [5, 20]:
            df[f"market_momentum_{w}d"] = df.groupby("ticker")[
                "market_return_1d"
            ].transform(lambda x: x.rolling(w).mean())

        # Relative return: how much this ticker beats / lags the market today
        df["relative_return_1d"] = df["return_1d"] - df["market_return_1d"]
        # Relative 5-day return vs rolling 5-day market average
        df["relative_return_5d"] = df["return_5d"] - df.groupby("ticker")[
            "market_return_1d"
        ].transform(lambda x: x.rolling(5).mean())

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of computed feature column names."""
        return self.feature_cols.copy()
