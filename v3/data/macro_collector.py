"""Cross-asset macro data collector (yfinance + FRED).

Sources:
  - yfinance: VIX, VIX3M, DXY, Gold, HYG, TLT (free, no key)
  - FRED: DGS10, DGS2, BAMLH0A0HYM2 (HY OAS) — needs FRED_API_KEY
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from loguru import logger


YFINANCE_SERIES = {
    "VIX":    "^VIX",
    "VIX3M":  "^VIX3M",
    "DXY":    "DX-Y.NYB",
    "GOLD":   "GC=F",
    "HYG":    "HYG",
    "TLT":    "TLT",
    "SPY":    "SPY",
}

FRED_SERIES = {
    "DGS10":        "DGS10",          # 10Y UST
    "DGS2":         "DGS2",           # 2Y UST
    "HY_OAS":       "BAMLH0A0HYM2",   # HY OAS
}


class MacroCollector:
    """Collects cross-asset macro indicators with parquet caching."""

    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        save_dir: str = "v3/data/raw",
        history_years: int = 5,
        fred_api_key: str | None = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history_years = history_years
        # NOTE: do NOT cache `datetime.now()` here — see OHLCVCollector. The
        # long-running daemon would freeze the macro data clock at boot time,
        # staling regime detection (2026-05-20 bug). `end`/`start` are computed
        # fresh per collect() call.
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY", "")

        self.out_path = self.save_dir / "macro.parquet"

    def collect(self, incremental_days: int | None = None) -> pd.DataFrame:
        """Collect all macro series. Returns wide DataFrame indexed by date."""
        end = datetime.now()
        start = (
            end - timedelta(days=incremental_days)
            if incremental_days
            else end - timedelta(days=self.history_years * 365)
        )

        frames: list[pd.Series] = []

        # yfinance series
        for name, ticker in YFINANCE_SERIES.items():
            try:
                s = self._fetch_yfinance(ticker, start, end)
                if s is not None and len(s) > 0:
                    s.name = name
                    frames.append(s)
                    logger.info(f"  yfinance {name} ({ticker}): {len(s)} rows")
                else:
                    logger.warning(f"  yfinance {name} ({ticker}): empty")
            except Exception as e:
                logger.warning(f"  yfinance {name} failed: {e}")

        # FRED series
        if self.fred_api_key:
            for name, sid in FRED_SERIES.items():
                try:
                    s = self._fetch_fred(sid, start, end)
                    if s is not None and len(s) > 0:
                        s.name = name
                        frames.append(s)
                        logger.info(f"  FRED {name} ({sid}): {len(s)} rows")
                    else:
                        logger.warning(f"  FRED {name} ({sid}): empty")
                except Exception as e:
                    logger.warning(f"  FRED {name} failed: {e}")
        else:
            logger.warning("FRED_API_KEY missing — yield curve & credit spread disabled")

        if not frames:
            logger.error("No macro data collected")
            return pd.DataFrame()

        # Align on daily index, forward-fill (macro data has gaps)
        df = pd.concat(frames, axis=1)
        df.index.name = "date"
        df = df.sort_index().ffill(limit=5)  # Limit ffill to 5 days (stale data)

        # Merge with existing cache if incremental
        if incremental_days and self.out_path.exists():
            existing = pd.read_parquet(self.out_path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        df.to_parquet(self.out_path, compression="snappy")
        logger.info(f"Macro saved: {df.shape} → {self.out_path}")
        return df

    def load(self) -> pd.DataFrame | None:
        """Load cached macro data."""
        if self.out_path.exists():
            df = pd.read_parquet(self.out_path)
            logger.info(f"Macro loaded: {df.shape}")
            return df
        return None

    def _fetch_yfinance(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.Series | None:
        """Fetch single yfinance series, return close price."""
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return None

        # yfinance may return MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = data["Close"]

        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        return close.astype(float)

    def _fetch_fred(
        self, series_id: str, start: datetime, end: datetime
    ) -> pd.Series | None:
        """Fetch single FRED series via REST API."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        r = requests.get(self.FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("observations", [])
        if not data:
            return None

        idx, vals = [], []
        for obs in data:
            v = obs.get("value", ".")
            if v == "." or v == "":
                continue
            idx.append(pd.Timestamp(obs["date"]).normalize())
            vals.append(float(v))

        if not vals:
            return None
        return pd.Series(vals, index=pd.DatetimeIndex(idx))
