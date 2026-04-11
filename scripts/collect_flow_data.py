"""Collect institutional/foreign investor flow data for KRX tickers.

Run this ONCE before training to build flow_data.parquet.
Subsequent runs skip already-collected tickers (incremental).

Usage:
    python scripts/collect_flow_data.py [--years 2] [--resume]

Data source: Naver Finance (HTML scraping)
Output: data/raw/flow_data.parquet
Time estimate: ~0.5s per page, 25 pages per ticker for 2 years
  238 tickers × 25 pages × 0.5s ≈ 50 minutes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from config.config_loader import load_config
from data.flow_data import FlowDataCollector


def main():
    parser = argparse.ArgumentParser(description="Collect flow data from Naver Finance")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history to collect")
    parser.add_argument("--resume", action="store_true", help="Skip tickers already in existing parquet")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_data"])
    save_path = str(raw_dir / "flow_data.parquet")

    # Load processed data to get KRX ticker list and date range
    processed_path = Path(cfg["paths"]["processed_data"]) / "processed_data.parquet"
    if not processed_path.exists():
        logger.error(f"processed_data.parquet not found at {processed_path}")
        logger.error("Run data collection + processing first")
        sys.exit(1)

    df = pd.read_parquet(processed_path, columns=["ticker", "date"])
    krx_tickers = sorted(
        df[df["ticker"].str.contains(r"\.(KS|KQ)$", na=False)]["ticker"].unique()
    )
    logger.info(f"KRX tickers to collect: {len(krx_tickers)}")

    # Date range
    end_date = df["date"].max()
    start_date = end_date - pd.Timedelta(days=int(args.years * 365))
    logger.info(f"Collection period: {start_date.date()} ~ {end_date.date()}")

    # Resume: skip already collected tickers
    if args.resume and Path(save_path).exists():
        existing = pd.read_parquet(save_path)
        existing_tickers = set(existing["ticker"].unique())
        remaining = [t for t in krx_tickers if t not in existing_tickers]
        logger.info(
            f"Resume mode: {len(existing_tickers)} tickers already collected, "
            f"{len(remaining)} remaining"
        )
        krx_tickers = remaining

    if not krx_tickers:
        logger.info("All tickers already collected. Done.")
        return

    # Collect
    collector = FlowDataCollector()
    flow_df = collector.collect_flow_data(
        tickers=krx_tickers,
        start_date=str(start_date.date()),
        end_date=str(end_date.date()),
        market="KOSPI",  # Naver works for both KOSPI and KOSDAQ
        save_path=save_path,
    )

    # If resuming, merge with existing data
    if args.resume and Path(save_path).exists():
        try:
            existing = pd.read_parquet(save_path)
            # The collector already saved, but we need to merge old + new
            if not flow_df.empty:
                merged = pd.concat([existing, flow_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date", "ticker"]).sort_values(
                    ["ticker", "date"]
                )
                merged.to_parquet(save_path, engine="pyarrow", compression="snappy")
                logger.info(f"Merged: {len(merged):,} total rows")
        except Exception as e:
            logger.warning(f"Merge with existing failed: {e}")

    if flow_df is not None and not flow_df.empty:
        logger.info(f"\nSummary:")
        logger.info(f"  Rows: {len(flow_df):,}")
        logger.info(f"  Tickers: {flow_df['ticker'].nunique()}")
        logger.info(f"  Date range: {flow_df['date'].min().date()} ~ {flow_df['date'].max().date()}")
        logger.info(f"  Saved to: {save_path}")
    else:
        logger.warning("No data collected")


if __name__ == "__main__":
    main()
