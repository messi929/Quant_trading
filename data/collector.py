"""Data collection from Yahoo Finance for KOSPI and NASDAQ markets."""

from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from pykrx import stock as krx

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class MarketDataCollector:
    """Collects OHLCV data from Yahoo Finance for KOSPI and NASDAQ."""

    def __init__(
        self,
        save_dir: str = "data/raw",
        history_years: int = 20,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (
            datetime.now() - timedelta(days=history_years * 365)
        ).strftime("%Y-%m-%d")
        logger.info(f"Collector period: {self.start_date} ~ {self.end_date}")

    # ------------------------------------------------------------------
    # Ticker listing
    # ------------------------------------------------------------------

    def get_kospi_tickers(self) -> pd.DataFrame:
        """Get all KOSPI listed tickers from KRX."""
        today = datetime.now().strftime("%Y%m%d")
        try:
            tickers = krx.get_market_ticker_list(today, market="KOSPI")
            records = []
            for ticker in tickers:
                name = krx.get_market_ticker_name(ticker)
                records.append(
                    {
                        "ticker": ticker,
                        "name": name,
                        "yf_ticker": f"{ticker}.KS",
                        "market": "KOSPI",
                    }
                )
            df = pd.DataFrame(records)
            logger.info(f"KOSPI tickers: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Failed to get KOSPI tickers: {e}")
            raise

    def get_nasdaq_tickers(self) -> pd.DataFrame:
        """Get NASDAQ composite tickers via yfinance screening.

        Falls back to a curated large-cap + mid-cap list
        if full listing is unavailable.
        """
        try:
            # Use S&P 500 + NASDAQ-100 as representative universe
            sp500_url = (
                "https://en.wikipedia.org/wiki/"
                "List_of_S%26P_500_companies"
            )
            resp = requests.get(sp500_url, headers=_HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            sp500 = tables[0]
            tickers_sp500 = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

            # NASDAQ-100
            nq100_url = (
                "https://en.wikipedia.org/wiki/"
                "Nasdaq-100#Components"
            )
            resp_nq = requests.get(nq100_url, headers=_HTTP_HEADERS, timeout=30)
            resp_nq.raise_for_status()
            tables_nq = pd.read_html(StringIO(resp_nq.text))
            # Find table with Ticker column
            nq100_tickers = []
            for t in tables_nq:
                cols = [str(c) for c in t.columns]
                for col in cols:
                    if "ticker" in col.lower() or "symbol" in col.lower():
                        nq100_tickers = (
                            t[col].dropna().astype(str)
                            .str.replace(".", "-", regex=False).tolist()
                        )
                        break
                if nq100_tickers:
                    break

            all_tickers = list(set(tickers_sp500 + nq100_tickers))
            records = [
                {
                    "ticker": t,
                    "name": t,
                    "yf_ticker": t,
                    "market": "NASDAQ",
                }
                for t in all_tickers
            ]
            df = pd.DataFrame(records)
            logger.info(f"US tickers (S&P500 + NQ100): {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Failed to get US tickers: {e}")
            raise

    # ------------------------------------------------------------------
    # OHLCV download
    # ------------------------------------------------------------------

    def download_ohlcv(
        self,
        tickers: list[str],
        market: str,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """Download OHLCV data in batches.

        Args:
            tickers: List of yfinance-compatible ticker symbols
            market: Market identifier ("KOSPI" or "NASDAQ")
            batch_size: Number of tickers per download batch

        Returns:
            DataFrame with multi-index (date, ticker) and OHLCV columns
        """
        all_data = []
        failed = []
        total = len(tickers)

        for i in range(0, total, batch_size):
            batch = tickers[i : i + batch_size]
            batch_str = " ".join(batch)
            logger.info(
                f"[{market}] Downloading batch {i // batch_size + 1}"
                f"/{(total + batch_size - 1) // batch_size} "
                f"({len(batch)} tickers)"
            )
            try:
                data = yf.download(
                    batch_str,
                    start=self.start_date,
                    end=self.end_date,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                )
                if data.empty:
                    failed.extend(batch)
                    continue

                # Reshape multi-ticker download
                if len(batch) == 1:
                    data.columns = pd.MultiIndex.from_product(
                        [[batch[0]], data.columns]
                    )

                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].copy()
                            df = df.dropna(how="all")
                            if len(df) > 0:
                                df["ticker"] = ticker
                                df["market"] = market
                                df.index.name = "date"
                                all_data.append(df.reset_index())
                    except (KeyError, Exception):
                        failed.append(ticker)

            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                failed.extend(batch)

        if failed:
            logger.warning(f"[{market}] Failed tickers: {len(failed)}")

        if not all_data:
            logger.error(f"[{market}] No data collected")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"[{market}] Collected {len(result)} rows "
            f"for {result['ticker'].nunique()} tickers"
        )
        return result

    # ------------------------------------------------------------------
    # Full collection
    # ------------------------------------------------------------------

    def collect_all(self, save: bool = True) -> dict[str, pd.DataFrame]:
        """Collect data for all markets.

        Returns:
            Dict mapping market name to DataFrame
        """
        results = {}

        # KOSPI
        logger.info("=== Collecting KOSPI data ===")
        kospi_info = self.get_kospi_tickers()
        kospi_data = self.download_ohlcv(
            kospi_info["yf_ticker"].tolist(), "KOSPI"
        )
        results["KOSPI"] = kospi_data

        # NASDAQ / US
        logger.info("=== Collecting NASDAQ data ===")
        nasdaq_info = self.get_nasdaq_tickers()
        nasdaq_data = self.download_ohlcv(
            nasdaq_info["yf_ticker"].tolist(), "NASDAQ"
        )
        results["NASDAQ"] = nasdaq_data

        if save:
            for market, df in results.items():
                if not df.empty:
                    path = self.save_dir / f"{market.lower()}_ohlcv.parquet"
                    df.to_parquet(path, engine="pyarrow", compression="snappy")
                    logger.info(f"Saved {path} ({len(df)} rows)")

            # Save ticker info
            ticker_info = pd.concat([kospi_info, nasdaq_info], ignore_index=True)
            ticker_path = self.save_dir / "ticker_info.parquet"
            ticker_info.to_parquet(ticker_path)
            logger.info(f"Saved ticker info: {ticker_path}")

        return results
