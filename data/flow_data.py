"""Institutional and foreign investor flow data from KRX via pykrx.

Collects net buying/selling data for KOSPI tickers to capture
non-OHLCV signals that retail-focused models miss.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from pykrx import stock as krx


class FlowDataCollector:
    """Collects institutional/foreign investor flow data from KRX.

    Features generated:
    - foreign_net_buy: 외국인 순매수 금액
    - inst_net_buy: 기관 순매수 금액
    - foreign_net_ratio: 외국인 순매수 / 거래대금 비율
    - inst_net_ratio: 기관 순매수 / 거래대금 비율
    - foreign_cumulative_5d: 외국인 5일 누적 순매수
    - inst_cumulative_5d: 기관 5일 누적 순매수
    - flow_momentum_*d: 수급 모멘텀 (rolling sum)
    - flow_divergence: 외국인-기관 방향성 차이
    """

    WINDOWS = [5, 10, 20]

    def collect_flow_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        market: str = "KOSPI",
    ) -> pd.DataFrame:
        """Collect net buying data for given tickers.

        Args:
            tickers: KRX ticker codes (e.g., ["005930", "000660"])
            start_date: "YYYYMMDD" format
            end_date: "YYYYMMDD" format
            market: Market identifier for logging

        Returns:
            DataFrame with columns: date, ticker, foreign_net_buy, inst_net_buy, trading_value
        """
        if market not in ("KOSPI", "KOSDAQ"):
            logger.debug(f"Flow data only available for KRX markets, skipping {market}")
            return pd.DataFrame()

        # Build mapping: clean_code -> original_ticker (preserving .KS/.KQ suffix)
        ticker_map = {}
        for t in tickers:
            clean = t.replace(".KS", "").replace(".KQ", "")
            ticker_map[clean] = t  # preserve original suffix
        clean_tickers = list(ticker_map.keys())
        all_records = []
        failed = 0

        for i, ticker in enumerate(clean_tickers):
            if i % 50 == 0:
                logger.info(f"[Flow] Collecting {i+1}/{len(clean_tickers)} tickers")
            try:
                # pykrx: 투자자별 순매수 (거래대금 기준)
                flow = krx.get_market_trading_value_by_date(
                    start_date, end_date, ticker
                )
                if flow is None or flow.empty:
                    continue

                for date_idx, row in flow.iterrows():
                    date_str = date_idx if isinstance(date_idx, str) else date_idx.strftime("%Y-%m-%d")
                    # pykrx columns: 기관합계, 외국인합계, 개인, etc.
                    # 순매수 = 매수 - 매도 (net buy positive)
                    foreign_net = 0.0
                    inst_net = 0.0
                    trading_val = 0.0

                    # Different pykrx versions use different column names
                    for col in row.index:
                        col_str = str(col)
                        if "외국인" in col_str:
                            foreign_net = float(row[col])
                        elif "기관" in col_str and "합계" in col_str:
                            inst_net = float(row[col])
                        elif "기관" in col_str and "합계" not in col_str:
                            # If no "합계", accumulate individual institutional investors
                            if inst_net == 0:
                                inst_net = float(row[col])
                        elif "전체" in col_str or "거래대금" in col_str:
                            trading_val = abs(float(row[col]))

                    # Estimate trading value from foreign + inst + retail if not available
                    if trading_val == 0:
                        trading_val = sum(abs(float(row[c])) for c in row.index) / 2

                    all_records.append({
                        "date": date_str,
                        "ticker": ticker_map.get(ticker, f"{ticker}.KS"),
                        "foreign_net_buy": foreign_net,
                        "inst_net_buy": inst_net,
                        "trading_value": max(trading_val, 1.0),
                    })

            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.debug(f"Flow data failed for {ticker}: {e}")
                continue

        if not all_records:
            logger.warning("No flow data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(
            f"[Flow] Collected {len(df):,} rows for {df['ticker'].nunique()} tickers "
            f"({failed} failed)"
        )
        return df


class FlowFeatureEngineer:
    """Computes flow-based features from institutional/foreign net buying data."""

    WINDOWS = [5, 10, 20]

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flow features to existing OHLCV dataframe.

        Args:
            df: DataFrame with at least date, ticker, volume, close columns
                AND merged flow data (foreign_net_buy, inst_net_buy, trading_value)

        Returns:
            DataFrame with additional flow feature columns
        """
        if "foreign_net_buy" not in df.columns:
            logger.debug("No flow data columns found — skipping flow features")
            return df

        df = df.sort_values(["ticker", "date"]).copy()

        # Base ratios (scaled by trading value for cross-sectional comparability)
        tv = df["trading_value"].clip(lower=1.0)
        df["foreign_net_ratio"] = (df["foreign_net_buy"] / tv).clip(-5, 5)
        df["inst_net_ratio"] = (df["inst_net_buy"] / tv).clip(-5, 5)

        # Flow features per ticker
        groups = df.groupby("ticker", group_keys=False)

        for w in self.WINDOWS:
            # Cumulative net buying over window
            df[f"foreign_cumul_{w}d"] = groups["foreign_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
            df[f"inst_cumul_{w}d"] = groups["inst_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
            # Flow momentum (change in cumulative)
            df[f"flow_momentum_{w}d"] = groups["foreign_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ) + groups["inst_net_ratio"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

        # Flow divergence: foreign vs institutional direction
        df["flow_divergence"] = np.sign(df["foreign_net_ratio"]) - np.sign(df["inst_net_ratio"])

        # Foreign buying intensity: standardized within rolling window
        df["foreign_intensity_20d"] = groups["foreign_net_ratio"].transform(
            lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-8)
        ).clip(-5, 5)

        # Cross-sectional rank of flow (per date)
        for feat in ["foreign_net_ratio", "inst_net_ratio"]:
            df[f"{feat}_rank"] = df.groupby("date")[feat].rank(pct=True)

        logger.info(f"Flow features computed: {len(self.get_feature_names())} features")
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of flow feature column names."""
        names = [
            "foreign_net_ratio", "inst_net_ratio",
            "flow_divergence", "foreign_intensity_20d",
            "foreign_net_ratio_rank", "inst_net_ratio_rank",
        ]
        for w in self.WINDOWS:
            names.extend([
                f"foreign_cumul_{w}d",
                f"inst_cumul_{w}d",
                f"flow_momentum_{w}d",
            ])
        return names
