"""Institutional and foreign investor flow data via Naver Finance.

Collects daily net buying/selling data (shares) for KRX tickers.
pykrx investor API is broken (KRX API change, 2026-04), so we use
Naver Finance HTML scraping as a reliable alternative.

Data source: https://finance.naver.com/item/frgn.naver?code={ticker}
- 외국인 순매수 (주수)
- 기관 순매수 (주수)
- 거래량
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from io import StringIO
from loguru import logger


class FlowDataCollector:
    """Collects institutional/foreign investor flow data from Naver Finance.

    Features generated:
    - foreign_net_buy: 외국인 순매수 (주수)
    - inst_net_buy: 기관 순매수 (주수)
    - trading_value: 거래량 (주수) — close × volume으로 거래대금 추정용
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    # Rate limit: Naver allows ~2 requests/sec safely
    REQUEST_DELAY = 0.5

    def collect_flow_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        market: str = "KOSPI",
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Collect net buying data for given tickers from Naver Finance.

        Args:
            tickers: KRX ticker codes with suffix (e.g., ["005930.KS", "000660.KS"])
            start_date: "YYYY-MM-DD" or "YYYYMMDD" format
            end_date: "YYYY-MM-DD" or "YYYYMMDD" format
            market: Market identifier ("KOSPI" or "KOSDAQ")
            save_path: Optional path to save intermediate results

        Returns:
            DataFrame with columns: date, ticker, foreign_net_buy, inst_net_buy, volume
        """
        if market not in ("KOSPI", "KOSDAQ"):
            logger.debug(f"Flow data only available for KRX markets, skipping {market}")
            return pd.DataFrame()

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_records = []
        failed = 0
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            # Extract 6-digit code
            code = ticker.replace(".KS", "").replace(".KQ", "")
            if not code.isdigit() or len(code) != 6:
                continue

            if i % 20 == 0:
                logger.info(
                    f"[Flow] Collecting {i+1}/{total} tickers "
                    f"({len(all_records):,} rows so far)"
                )

            try:
                records = self._fetch_ticker_flow(
                    code, ticker, start_dt, end_dt
                )
                all_records.extend(records)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.debug(f"Flow data failed for {code}: {e}")

            # Save intermediate results every 50 tickers
            if save_path and i > 0 and i % 50 == 0:
                self._save_intermediate(all_records, save_path)

        if not all_records:
            logger.warning("No flow data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info(
            f"[Flow] Collected {len(df):,} rows for {df['ticker'].nunique()} tickers "
            f"({failed} failed)"
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, engine="pyarrow", compression="snappy")
            logger.info(f"[Flow] Saved to {save_path}")

        return df

    def _fetch_ticker_flow(
        self,
        code: str,
        original_ticker: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> list[dict]:
        """Fetch flow data for a single ticker from Naver Finance."""
        records = []
        page = 1
        max_pages = 200  # Safety limit (~4000 trading days)
        consecutive_old = 0

        while page <= max_pages:
            time.sleep(self.REQUEST_DELAY)

            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}"
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                if resp.status_code != 200:
                    break

                dfs = pd.read_html(StringIO(resp.text))
                if len(dfs) < 2:
                    break

                # Find the investor trading table (MultiIndex, 9 columns)
                df = None
                for candidate in dfs:
                    if candidate.shape[1] == 9 and isinstance(
                        candidate.columns, pd.MultiIndex
                    ):
                        df = candidate
                        break
                if df is None:
                    break

                # Flatten multi-index columns
                df.columns = [f"{a}_{b}" for a, b in df.columns]

                # Filter valid date rows
                date_col = df.columns[0]
                df = df.dropna(subset=[date_col])
                df = df[
                    df[date_col]
                    .astype(str)
                    .str.match(r"\d{4}\.\d{2}\.\d{2}", na=False)
                ]

                if df.empty:
                    break

                # Parse dates
                dates = pd.to_datetime(df[date_col])
                oldest_date = dates.min()

                # Stop if we've gone past start_date
                if oldest_date < start_dt:
                    consecutive_old += 1
                    # Still process this page — some rows may be in range
                else:
                    consecutive_old = 0

                # Columns: date, close, change, change_pct, volume, inst_net, foreign_net, foreign_shares, foreign_pct
                cols = df.columns.tolist()
                for _, row in df.iterrows():
                    try:
                        row_date = pd.to_datetime(row[cols[0]])
                        if row_date < start_dt or row_date > end_dt:
                            continue

                        volume = self._parse_number(row[cols[4]])
                        inst_net = self._parse_number(row[cols[5]])
                        foreign_net = self._parse_number(row[cols[6]])

                        records.append({
                            "date": row_date,
                            "ticker": original_ticker,
                            "foreign_net_buy": foreign_net,
                            "inst_net_buy": inst_net,
                            "volume": volume,
                        })
                    except (ValueError, IndexError):
                        continue

                # Stop conditions
                if consecutive_old >= 2:
                    break
                if oldest_date < start_dt:
                    break

                page += 1

            except requests.RequestException:
                break
            except Exception:
                break

        return records

    @staticmethod
    def _parse_number(val) -> float:
        """Parse number from Naver Finance format (handles commas, signs)."""
        if pd.isna(val):
            return 0.0
        s = str(val).replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _save_intermediate(records: list[dict], path: str) -> None:
        """Save intermediate results to avoid data loss on interruption."""
        try:
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, engine="pyarrow", compression="snappy")
        except Exception as e:
            logger.debug(f"Intermediate save failed: {e}")


class FlowFeatureEngineer:
    """Computes flow-based features from institutional/foreign net buying data.

    Input requires: foreign_net_buy, inst_net_buy columns (shares, not monetary value).
    Uses volume for normalization (net_buy / volume → net_ratio).
    """

    WINDOWS = [5, 10, 20]

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flow features to existing OHLCV dataframe.

        Args:
            df: DataFrame with at least date, ticker, volume, close columns
                AND merged flow data (foreign_net_buy, inst_net_buy)

        Returns:
            DataFrame with additional flow feature columns
        """
        if "foreign_net_buy" not in df.columns:
            logger.debug("No flow data columns found — skipping flow features")
            return df

        df = df.sort_values(["ticker", "date"]).copy()

        # Base ratios: net buy shares / volume (cross-sectional comparability)
        vol_safe = df["volume"].clip(lower=1.0)
        df["foreign_net_ratio"] = (df["foreign_net_buy"] / vol_safe).clip(-5, 5)
        df["inst_net_ratio"] = (df["inst_net_buy"] / vol_safe).clip(-5, 5)

        # Flow features per ticker
        groups = df.groupby("ticker", group_keys=False)

        for w in self.WINDOWS:
            # Cumulative net buying ratio over window
            df[f"foreign_cumul_{w}d"] = groups["foreign_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
            df[f"inst_cumul_{w}d"] = groups["inst_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
            # Flow momentum: average combined net ratio
            df[f"flow_momentum_{w}d"] = groups["foreign_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ) + groups["inst_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

        # Flow divergence: foreign vs institutional direction
        df["flow_divergence"] = (
            np.sign(df["foreign_net_ratio"]) - np.sign(df["inst_net_ratio"])
        )

        # Foreign buying intensity: z-score within rolling window
        df["foreign_intensity_20d"] = groups["foreign_net_ratio"].transform(
            lambda x: (
                (x - x.rolling(20, min_periods=5).mean())
                / (x.rolling(20, min_periods=5).std() + 1e-8)
            )
        ).clip(-5, 5)

        # Cross-sectional rank of flow (per date)
        for feat in ["foreign_net_ratio", "inst_net_ratio"]:
            df[f"{feat}_rank"] = df.groupby("date")[feat].rank(pct=True)

        n_features = len(self.get_feature_names())
        logger.info(f"Flow features computed: {n_features} features")
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of flow feature column names."""
        names = [
            "foreign_net_ratio",
            "inst_net_ratio",
            "flow_divergence",
            "foreign_intensity_20d",
            "foreign_net_ratio_rank",
            "inst_net_ratio_rank",
        ]
        for w in self.WINDOWS:
            names.extend([
                f"foreign_cumul_{w}d",
                f"inst_cumul_{w}d",
                f"flow_momentum_{w}d",
            ])
        return names
