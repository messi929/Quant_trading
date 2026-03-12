"""DART 공시 이벤트 수집 (금융감독원 전자공시시스템).

DART OpenAPI를 통해 주요 공시를 실시간 수집하고,
이벤트 기반 신호로 변환합니다.

API 키: https://opendart.fss.or.kr 에서 무료 발급
환경변수: DART_API_KEY
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from loguru import logger


DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 공시 유형별 중요도 가중치
DISCLOSURE_WEIGHTS = {
    # 긍정적 이벤트
    "자기주식취득결정": 0.8,
    "배당결정": 0.5,
    "유상증자결정": -0.3,  # 희석 효과
    "무상증자결정": 0.4,
    "주식분할결정": 0.3,
    "자기주식처분결정": -0.4,
    "합병결정": 0.2,
    # 실적 관련
    "매출액또는손익구조": 0.6,  # 방향은 내용 분석 필요
    "영업실적": 0.6,
    # 부정적 이벤트
    "상장폐지": -1.0,
    "관리종목지정": -0.8,
    "감사의견거절": -0.9,
    "횡령배임": -0.7,
    "소송": -0.3,
}

# 주요 공시 유형 코드 (DART report_tp)
REPORT_TYPES = {
    "A": "사업보고서",
    "B": "반기보고서",
    "C": "분기보고서",
    "D": "등록법인공시",
    "E": "기타공시",
    "F": "외부감사관련",
    "G": "펀드공시",
    "H": "자산유동화",
    "I": "거래소공시",
    "J": "공정위공시",
}


class DartClient:
    """DART OpenAPI 클라이언트.

    주요 기능:
    - 최근 공시 목록 조회
    - 공시 내용 기반 이벤트 신호 생성
    - 종목별 공시 이력 추적
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DART_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "DART_API_KEY 미설정 — 공시 이벤트 기능 비활성화. "
                "https://opendart.fss.or.kr 에서 API 키 발급 후 "
                ".env에 DART_API_KEY=xxx 추가"
            )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "QuantTrading/1.0"})

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_recent_disclosures(
        self,
        days_back: int = 1,
        corp_code: str = None,
    ) -> pd.DataFrame:
        """최근 공시 목록 조회.

        Args:
            days_back: 몇 일 전까지 조회할지
            corp_code: 특정 기업 코드 (None이면 전체)

        Returns:
            DataFrame with columns: corp_name, corp_code, report_nm,
            rcept_dt, flr_nm, report_tp
        """
        if not self.is_available:
            return pd.DataFrame()

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")

        params = {
            "crtfc_key": self.api_key,
            "bgn_de": start_date,
            "end_de": end_date,
            "page_count": 100,
            "sort": "date",
            "sort_mth": "desc",
        }
        if corp_code:
            params["corp_code"] = corp_code

        try:
            resp = self._session.get(
                f"{DART_BASE_URL}/list.json",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "000":
                logger.debug(f"DART API: {data.get('message', 'unknown error')}")
                return pd.DataFrame()

            items = data.get("list", [])
            if not items:
                return pd.DataFrame()

            df = pd.DataFrame(items)
            logger.info(f"DART 공시 {len(df)}건 조회 ({start_date}~{end_date})")
            return df

        except Exception as e:
            logger.warning(f"DART API 호출 실패: {e}")
            return pd.DataFrame()

    def score_disclosure(self, report_name: str) -> float:
        """공시 제목 기반 이벤트 스코어 산출.

        Args:
            report_name: 공시 보고서명

        Returns:
            -1.0 ~ +1.0 이벤트 스코어 (0=중립)
        """
        if not isinstance(report_name, str):
            return 0.0

        score = 0.0
        matched = False
        for keyword, weight in DISCLOSURE_WEIGHTS.items():
            if keyword in report_name:
                score += weight
                matched = True

        # 미매칭: 중립
        if not matched:
            return 0.0

        return max(-1.0, min(1.0, score))

    def get_event_signals(
        self,
        ticker_map: dict = None,
        days_back: int = 1,
    ) -> dict:
        """공시 기반 이벤트 신호 생성.

        Args:
            ticker_map: {corp_name: ticker_code} 매핑 (None이면 이름 그대로)
            days_back: 조회 기간

        Returns:
            {ticker: {"score": float, "events": list[str], "count": int}}
        """
        disclosures = self.get_recent_disclosures(days_back=days_back)
        if disclosures.empty:
            return {}

        signals = {}
        for _, row in disclosures.iterrows():
            corp_name = row.get("corp_name", "")
            report_nm = row.get("report_nm", "")
            score = self.score_disclosure(report_nm)

            if abs(score) < 0.1:
                continue  # 중립 공시 스킵

            # 종목 코드 매핑
            ticker = corp_name
            if ticker_map and corp_name in ticker_map:
                ticker = ticker_map[corp_name]

            if ticker not in signals:
                signals[ticker] = {"score": 0.0, "events": [], "count": 0}

            signals[ticker]["score"] += score
            signals[ticker]["events"].append(f"{report_nm} ({score:+.1f})")
            signals[ticker]["count"] += 1

        # 스코어 클리핑
        for ticker in signals:
            signals[ticker]["score"] = max(-1.0, min(1.0, signals[ticker]["score"]))

        if signals:
            logger.info(
                f"DART 이벤트 신호: {len(signals)}종목 "
                f"(양={sum(1 for s in signals.values() if s['score']>0)}, "
                f"음={sum(1 for s in signals.values() if s['score']<0)})"
            )

        return signals
