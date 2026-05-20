"""OHLCV data collection from yfinance with batch processing."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from v3.data.universe import Universe, TickerInfo


class OHLCVCollector:
    """Downloads OHLCV data for all tickers in the universe."""

    def __init__(
        self,
        save_dir: str = "v3/data/raw",
        history_years: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history_years = history_years
        # NOTE: do NOT cache `datetime.now()` here. The live daemon constructs
        # this collector once at startup; caching end_date freezes the data
        # clock at boot time (2026-05-20 bug → repeated stale MU entries).
        # `end`/`start` are computed fresh per collect() call below.

    def collect(
        self,
        universe: Universe,
        incremental_days: int | None = None,
    ) -> pd.DataFrame:
        """Collect OHLCV for all tickers in universe.

        Args:
            universe: Ticker universe to collect data for.
            incremental_days: If set, only fetch last N days (for daily updates).

        Returns:
            Combined DataFrame with columns: date, open, high, low, close, volume, ticker, market.
        """
        end = datetime.now()
        if incremental_days:
            start = end - timedelta(days=incremental_days)
        else:
            start = end - timedelta(days=self.history_years * 365)

        all_frames = []

        for market in universe.markets:
            tickers = universe.ticker_list(market)
            if not tickers:
                continue

            logger.info(f"Collecting {market}: {len(tickers)} tickers "
                         f"({start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')})")

            df = self._download_batch(tickers, market, start, end)
            if df is not None and len(df) > 0:
                all_frames.append(df)

        if not all_frames:
            logger.warning("No data collected")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Save raw data
        path = self.save_dir / "ohlcv_raw.parquet"
        combined.to_parquet(path, compression="snappy")
        logger.info(f"Collected {len(combined)} rows, {combined['ticker'].nunique()} tickers → {path}")

        return combined

    def _download_batch(
        self,
        tickers: list[str],
        market: str,
        start: datetime,
        end: datetime,
        batch_size: int = 50,
    ) -> pd.DataFrame | None:
        """Download OHLCV in batches via yfinance."""
        all_records = []
        failed = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(f"  Batch {i // batch_size + 1}: {len(batch)} tickers")

            try:
                data = yf.download(
                    batch,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

                if data.empty:
                    failed.extend(batch)
                    continue

                records = self._reshape_yf_data(data, batch, market)
                all_records.extend(records)

            except Exception as e:
                logger.warning(f"  Batch download failed: {e}")
                failed.extend(batch)

        if failed:
            logger.warning(f"  Failed tickers ({len(failed)}): {failed[:5]}...")

        if not all_records:
            return None

        return pd.DataFrame(all_records)

    def _reshape_yf_data(
        self, data: pd.DataFrame, tickers: list[str], market: str
    ) -> list[dict]:
        """Reshape yfinance multi-ticker output to flat records."""
        records = []

        if len(tickers) == 1:
            ticker = tickers[0]
            for date, row in data.iterrows():
                if pd.isna(row.get("Close")):
                    continue
                records.append({
                    "date": pd.Timestamp(date).normalize(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    "ticker": ticker,
                    "market": market,
                })
        else:
            for ticker in tickers:
                try:
                    ticker_data = data.xs(ticker, level=1, axis=1) if isinstance(
                        data.columns, pd.MultiIndex
                    ) else data
                except (KeyError, TypeError):
                    continue

                for date, row in ticker_data.iterrows():
                    if pd.isna(row.get("Close")):
                        continue
                    records.append({
                        "date": pd.Timestamp(date).normalize(),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                        "ticker": ticker,
                        "market": market,
                    })

        return records

    def load_existing(self) -> pd.DataFrame | None:
        """Load previously collected raw data."""
        path = self.save_dir / "ohlcv_raw.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded existing data: {len(df)} rows")
            return df
        return None

    def merge_incremental(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """Merge incremental data with existing, deduplicating."""
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        path = self.save_dir / "ohlcv_raw.parquet"
        combined.to_parquet(path, compression="snappy")
        logger.info(f"Merged data: {len(combined)} rows → {path}")
        return combined
