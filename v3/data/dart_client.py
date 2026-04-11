"""DART corporate disclosure event collection and scoring."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger


# Disclosure event weights — positive = bullish, negative = bearish
DISCLOSURE_WEIGHTS: dict[str, float] = {
    # Positive
    "자기주식취득결정": 0.8,
    "자기주식처분결정": -0.3,
    "배당결정": 0.5,
    "주식분할결정": 0.3,
    "무상증자결정": 0.4,
    "합병결정": 0.2,
    "매출액또는손익구조": 0.3,    # 실적 공시 (방향 미확정)
    "영업실적": 0.3,
    # Negative
    "상장폐지": -1.0,
    "관리종목지정": -0.8,
    "감사의견거절": -0.9,
    "감사의견한정": -0.5,
    "횡령배임": -0.7,
    "유상증자결정": -0.3,
    "전환사채권발행결정": -0.2,
    "신주인수권부사채권발행결정": -0.2,
}


class DartClient:
    """DART OpenAPI wrapper for corporate disclosure events."""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("DART_API_KEY", "")
        self._available = bool(self.api_key)

    @property
    def is_available(self) -> bool:
        return self._available

    def get_recent_disclosures(
        self,
        days_back: int = 1,
        corp_code: str | None = None,
    ) -> pd.DataFrame:
        """Fetch recent disclosures from DART API.

        Returns:
            DataFrame with: corp_name, corp_code, report_nm, rcept_dt, report_tp.
        """
        if not self._available:
            logger.debug("DART API key not configured")
            return pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        params = {
            "crtfc_key": self.api_key,
            "bgn_de": start_date.strftime("%Y%m%d"),
            "end_de": end_date.strftime("%Y%m%d"),
            "page_count": 100,
        }
        if corp_code:
            params["corp_code"] = corp_code

        try:
            resp = requests.get(f"{self.BASE_URL}/list.json", params=params, timeout=10)
            data = resp.json()

            if data.get("status") != "000":
                logger.debug(f"DART API: {data.get('message', 'unknown error')}")
                return pd.DataFrame()

            items = data.get("list", [])
            return pd.DataFrame(items)

        except Exception as e:
            logger.warning(f"DART API error: {e}")
            return pd.DataFrame()

    def score_disclosure(self, report_name: str) -> float:
        """Score a disclosure by keyword matching. Returns [-1.0, 1.0]."""
        total = 0.0
        for keyword, weight in DISCLOSURE_WEIGHTS.items():
            if keyword in report_name:
                total += weight
        return max(-1.0, min(1.0, total))

    def get_event_signals(
        self,
        ticker_map: dict[str, str] | None = None,
        days_back: int = 5,
    ) -> dict[str, dict]:
        """Get event scores for tickers.

        Args:
            ticker_map: {corp_name: ticker} mapping.
            days_back: Number of days to look back.

        Returns:
            {ticker: {"score": float, "events": list[str], "count": int}}
        """
        df = self.get_recent_disclosures(days_back=days_back)
        if df.empty:
            return {}

        signals = {}
        for _, row in df.iterrows():
            corp_name = row.get("corp_name", "")
            report_name = row.get("report_nm", "")
            score = self.score_disclosure(report_name)

            if score == 0.0:
                continue

            # Find ticker from corp_name
            ticker = None
            if ticker_map and corp_name in ticker_map:
                ticker = ticker_map[corp_name]
            else:
                ticker = corp_name  # Fallback

            if ticker not in signals:
                signals[ticker] = {"score": 0.0, "events": [], "count": 0}

            signals[ticker]["score"] += score
            signals[ticker]["events"].append(report_name)
            signals[ticker]["count"] += 1

        # Clip scores
        for ticker in signals:
            signals[ticker]["score"] = max(-1.0, min(1.0, signals[ticker]["score"]))

        return signals
