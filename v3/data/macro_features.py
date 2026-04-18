"""Macro feature engineering — 7 cross-asset signals + rolling percentile.

Features:
  1. VIX term ratio (VIX / VIX3M) — stress indicator
  2. Yield curve slope (DGS10 - DGS2) — recession signal
  3. HY credit spread level + 60d change — risk-off indicator
  4. DXY 60d momentum — dollar strength (inverse risk-on)
  5. Gold/SPY ratio 60d momentum — flight to safety
  6. Breadth (% tickers above SMA50) — provided externally
  7. HYG/TLT ratio 60d momentum — credit risk appetite

All features are converted to rolling 5y percentile [0, 1] for composite scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class MacroFeatureEngineer:
    """Computes cross-asset features + percentile ranks."""

    PERCENTILE_WINDOW_DAYS = 1260   # 5 years rolling

    def __init__(self, breadth_ma_window: int = 50):
        self.breadth_ma = breadth_ma_window

    def compute(
        self,
        macro: pd.DataFrame,
        ohlcv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute all features. Returns DataFrame indexed by date.

        Args:
            macro: Wide macro DF from MacroCollector (columns: VIX, VIX3M, ...).
            ohlcv: V3 OHLCV DataFrame (for breadth). If None, breadth feature omitted.
        """
        feats = pd.DataFrame(index=macro.index)

        # 1. VIX term ratio (spot / 3M). >1 = stress
        if "VIX" in macro and "VIX3M" in macro:
            vix3m = macro["VIX3M"].replace(0, np.nan)
            feats["vix_ratio"] = macro["VIX"] / vix3m

        # 2. Yield curve slope (10Y - 2Y). <0 = recession signal
        if "DGS10" in macro and "DGS2" in macro:
            feats["yc_slope"] = macro["DGS10"] - macro["DGS2"]

        # 3. HY credit spread: level (absolute OAS) + 60d change
        if "HY_OAS" in macro:
            feats["hy_level"] = macro["HY_OAS"]
            feats["hy_change_60d"] = macro["HY_OAS"].diff(60)

        # 4. DXY 60d momentum — rising dollar = risk-off
        if "DXY" in macro:
            feats["dxy_mom_60d"] = macro["DXY"].pct_change(60, fill_method=None)

        # 5. Gold/SPY ratio 60d momentum — rising = flight to safety
        if "GOLD" in macro and "SPY" in macro:
            gold_spy = macro["GOLD"] / macro["SPY"].replace(0, np.nan)
            feats["gold_spy_mom_60d"] = gold_spy.pct_change(60, fill_method=None)

        # 6. HYG/TLT ratio 60d momentum — rising = credit risk appetite (risk-on)
        if "HYG" in macro and "TLT" in macro:
            hyg_tlt = macro["HYG"] / macro["TLT"].replace(0, np.nan)
            feats["hyg_tlt_mom_60d"] = hyg_tlt.pct_change(60, fill_method=None)

        # 7. Breadth — % of tickers above 50-day SMA
        if ohlcv is not None and len(ohlcv) > 0:
            breadth = self._compute_breadth(ohlcv)
            if breadth is not None:
                feats = feats.join(breadth.rename("breadth"), how="left")

        # Forward fill short gaps (macro has weekends/holidays; max 5d)
        feats = feats.ffill(limit=5)

        logger.info(f"Macro features computed: {feats.shape[1]} cols, "
                     f"coverage={feats.notna().mean().round(2).to_dict()}")
        return feats

    def compute_percentiles(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Compute 5-year rolling percentile rank for each feature."""
        pctl = pd.DataFrame(index=feats.index, columns=feats.columns, dtype=float)

        for col in feats.columns:
            s = feats[col].astype(float)
            pctl[col] = s.rolling(
                window=self.PERCENTILE_WINDOW_DAYS, min_periods=60
            ).apply(
                lambda x: (x[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5,
                raw=True,
            )

        return pctl.clip(0, 1)

    def _compute_breadth(self, ohlcv: pd.DataFrame) -> pd.Series | None:
        """% of tickers whose close > SMA50, daily."""
        if "ticker" not in ohlcv.columns or "close" not in ohlcv.columns:
            return None

        # Pivot to wide: rows=date, cols=ticker, values=close
        wide = ohlcv.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        ).sort_index()

        if wide.empty:
            return None

        sma50 = wide.rolling(self.breadth_ma, min_periods=20).mean()
        above = (wide > sma50).sum(axis=1)
        count = wide.notna().sum(axis=1).replace(0, np.nan)
        breadth = (above / count).clip(0, 1)
        breadth.index = pd.DatetimeIndex(breadth.index).normalize()
        return breadth
