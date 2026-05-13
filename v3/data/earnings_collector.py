"""Quarterly earnings date collector — yfinance get_earnings_dates().

Used by AlphaEarningsProximity (follow-up #2 candidate alpha). Collects
~20 most-recent earnings dates per ticker (covers ~5 years quarterly) and
caches to JSON for reproducibility.

Usage:
    PYTHONPATH=. python v3/data/earnings_collector.py \\
        --output v3/data/raw/earnings_dates.json

Output schema:
    {
      "metadata": {"collected_at": ISO, "n_tickers": N, "limit_per_ticker": 20},
      "data": {"AAPL": ["2026-04-30", "2026-01-29", ...], ...}
    }

Notes:
  - yfinance get_earnings_dates returns future + past dates; we keep only past
    (we cannot use future leakage in backtesting).
  - All dates normalized to date-only (drop intraday TZ component).
  - Tickers missing earnings_dates → empty list (filter downstream).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger


def collect_earnings_dates(
    tickers: list[str],
    limit_per_ticker: int = 20,
    sleep_sec: float = 0.5,
) -> dict[str, list[str]]:
    """Fetch earnings dates for each ticker.

    Returns {ticker: [ISO-date-str, ...]} sorted ascending (oldest first).
    """
    result: dict[str, list[str]] = {}
    today = pd.Timestamp.now(tz=None).normalize()

    for i, ticker in enumerate(tickers, 1):
        try:
            t = yf.Ticker(ticker)
            ed = t.get_earnings_dates(limit=limit_per_ticker)
            if ed is None or ed.empty:
                logger.warning(f"[{i}/{len(tickers)}] {ticker}: empty")
                result[ticker] = []
                continue
            # Index is timezone-aware Earnings Date; convert to naive date
            idx = ed.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None) if idx.tz is not None else idx
            dates = pd.to_datetime(idx).normalize()
            dates = sorted({d.to_pydatetime().date().isoformat() for d in dates})
            result[ticker] = dates
            logger.info(
                f"[{i}/{len(tickers)}] {ticker}: {len(dates)} dates "
                f"(range {dates[0]} ~ {dates[-1]})"
            )
        except (KeyError, ValueError, AttributeError, ConnectionError) as exc:
            logger.warning(f"[{i}/{len(tickers)}] {ticker}: {type(exc).__name__} {exc}")
            result[ticker] = []
        # Rate-limit safety
        if i < len(tickers):
            time.sleep(sleep_sec)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers-from",
        type=str,
        default="v3/data/raw/ohlcv_raw.parquet",
        help="Parquet path to extract ticker universe from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="v3/data/raw/earnings_dates.json",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_parquet(args.tickers_from)
    tickers = sorted(df["ticker"].unique().tolist())
    logger.info(f"Collecting earnings dates for {len(tickers)} tickers...")

    data = collect_earnings_dates(tickers, args.limit, args.sleep)

    n_with_data = sum(1 for v in data.values() if v)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "metadata": {
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "n_tickers": len(tickers),
            "n_with_data": n_with_data,
            "limit_per_ticker": args.limit,
            "source": "yfinance.Ticker.get_earnings_dates",
        },
        "data": data,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_path} ({n_with_data}/{len(tickers)} tickers populated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
