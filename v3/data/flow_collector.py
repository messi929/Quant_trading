"""Foreign/institutional flow data collection from Naver Finance."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger


class FlowDataCollector:
    """Collects foreign & institutional net buy/sell data from Naver Finance."""

    NAVER_URL = "https://finance.naver.com/item/frgn.naver"

    def __init__(self, save_dir: str = "v3/data/raw"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    def collect(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Collect flow data for domestic tickers.

        Args:
            tickers: List of KRX tickers ("005930.KS" format).
            start_date: Start date "YYYY-MM-DD".
            end_date: End date (default: today).

        Returns:
            DataFrame with: date, ticker, foreign_net_buy, inst_net_buy, volume.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_records = []

        for i, ticker in enumerate(tickers):
            code = ticker.split(".")[0]
            if not code.isdigit():
                continue  # Skip US tickers

            logger.info(f"  Flow [{i+1}/{len(tickers)}] {ticker}")

            try:
                records = self._fetch_ticker_flow(code, ticker, start_dt, end_dt)
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"  Failed {ticker}: {e}")

            # Rate limit
            time.sleep(0.5)

            # Intermediate save every 50 tickers
            if (i + 1) % 50 == 0 and all_records:
                self._save_intermediate(all_records)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])

        # Save
        path = self.save_dir / "flow_data.parquet"
        df.to_parquet(path, compression="snappy")
        logger.info(f"Flow data: {len(df)} rows, {df['ticker'].nunique()} tickers → {path}")

        return df

    def _fetch_ticker_flow(
        self, code: str, ticker: str, start_dt: datetime, end_dt: datetime
    ) -> list[dict]:
        """Fetch flow data for a single ticker via Naver Finance pagination."""
        records = []
        page = 1
        pages_past_start = 0

        while page <= 200:
            params = {"code": code, "page": page}
            try:
                resp = self.session.get(self.NAVER_URL, params=params, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                table = soup.select_one("table.type2")
                if not table:
                    break

                rows = table.select("tr")
                found_data = False

                for row in rows:
                    cols = row.select("td")
                    if len(cols) < 9:
                        continue

                    date_text = cols[0].get_text(strip=True)
                    if not date_text or "." not in date_text:
                        continue

                    try:
                        date = datetime.strptime(date_text, "%Y.%m.%d")
                    except ValueError:
                        continue

                    if date > end_dt:
                        continue
                    if date < start_dt:
                        pages_past_start += 1
                        if pages_past_start > 2:
                            return records
                        continue

                    found_data = True
                    volume = self._parse_number(cols[4].get_text(strip=True))
                    inst_net = self._parse_number(cols[5].get_text(strip=True))
                    foreign_net = self._parse_number(cols[6].get_text(strip=True))

                    records.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "foreign_net_buy": foreign_net,
                        "inst_net_buy": inst_net,
                        "volume": volume,
                    })

                if not found_data and page > 1:
                    break

                page += 1
                time.sleep(0.3)

            except Exception as e:
                logger.debug(f"Page {page} error for {code}: {e}")
                break

        return records

    @staticmethod
    def _parse_number(val: str) -> float:
        """Parse Korean number format: +1,234 → 1234.0"""
        val = val.replace(",", "").replace("+", "").strip()
        if not val or val == "-":
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    def _save_intermediate(self, records: list[dict]) -> None:
        df = pd.DataFrame(records)
        path = self.save_dir / "flow_data_partial.parquet"
        df.to_parquet(path, compression="snappy")

    def load_existing(self) -> pd.DataFrame | None:
        path = self.save_dir / "flow_data.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


class FlowFeatureEngineer:
    """Compute flow-based features from raw flow data."""

    FEATURE_NAMES = [
        "foreign_net_ratio",
        "inst_net_ratio",
        "foreign_cumul_5d",
        "foreign_cumul_10d",
        "foreign_cumul_20d",
        "inst_cumul_5d",
        "inst_cumul_10d",
        "inst_cumul_20d",
        "flow_momentum_5d",
        "flow_momentum_10d",
        "flow_divergence",
        "foreign_intensity_20d",
    ]

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Add flow features to DataFrame (must have flow columns merged)."""
        df = df.copy()

        # Base ratios (clipped)
        vol = df["volume"].replace(0, 1)
        df["foreign_net_ratio"] = (df["foreign_net_buy"] / vol).clip(-5, 5)
        df["inst_net_ratio"] = (df["inst_net_buy"] / vol).clip(-5, 5)

        # Cumulative flows per ticker
        for col_base, col_prefix in [
            ("foreign_net_ratio", "foreign_cumul"),
            ("inst_net_ratio", "inst_cumul"),
        ]:
            for w in [5, 10, 20]:
                df[f"{col_prefix}_{w}d"] = (
                    df.groupby("ticker")[col_base]
                    .transform(lambda x: x.rolling(w, min_periods=1).sum())
                )

        # Flow momentum
        combined = df["foreign_net_ratio"] + df["inst_net_ratio"]
        for w in [5, 10]:
            df[f"flow_momentum_{w}d"] = (
                df.groupby("ticker")[lambda _: combined]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            ) if False else combined.groupby(df["ticker"]).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

        # Flow divergence: foreign vs institutional agreement
        df["flow_divergence"] = (
            df["foreign_net_ratio"].apply(lambda x: 1 if x > 0 else -1)
            - df["inst_net_ratio"].apply(lambda x: 1 if x > 0 else -1)
        )

        # Foreign intensity z-score
        df["foreign_intensity_20d"] = (
            df.groupby("ticker")["foreign_net_ratio"]
            .transform(lambda x: (x - x.rolling(20, min_periods=5).mean())
                       / x.rolling(20, min_periods=5).std().replace(0, 1))
        ).clip(-5, 5)

        return df
