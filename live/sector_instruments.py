"""섹터 → 실제 투자 종목 매핑.

모델의 11개 GICS 섹터 신호를 실제 매수 가능한
국내/해외 ETF 또는 종목으로 변환합니다.
"""

import yaml
from pathlib import Path
from loguru import logger


# KOSPI 섹터 ETF 매핑 (KODEX 시리즈)
# 섹터 ETF가 없는 경우 가장 유사한 ETF로 대체
KOSPI_ETF_MAP = {
    "energy":                 {"ticker": "117460", "name": "KODEX 에너지화학",   "market": "domestic"},
    "materials":              {"ticker": "117460", "name": "KODEX 에너지화학",   "market": "domestic"},  # 소재 ETF 부재
    "industrials":            {"ticker": "214830", "name": "KODEX 건설",         "market": "domestic"},
    "consumer_discretionary": {"ticker": "091220", "name": "KODEX 소비재",       "market": "domestic"},
    "consumer_staples":       {"ticker": "091220", "name": "KODEX 소비재",       "market": "domestic"},  # 필수소비재 ETF 부재
    "healthcare":             {"ticker": "266410", "name": "KODEX 헬스케어",     "market": "domestic"},
    "financials":             {"ticker": "091160", "name": "KODEX 금융",         "market": "domestic"},
    "information_technology": {"ticker": "261110", "name": "KODEX IT",           "market": "domestic"},
    "communication_services": {"ticker": "261110", "name": "KODEX IT",           "market": "domestic"},  # 통신 ETF 부재
    "utilities":              {"ticker": "337140", "name": "KODEX 유틸리티",     "market": "domestic"},
    "real_estate":            {"ticker": "395400", "name": "KODEX 리츠부동산인프라", "market": "domestic"},
}

# NASDAQ 섹터 ETF 매핑 (XL 시리즈, 한국투자증권 해외주식)
NASDAQ_ETF_MAP = {
    "energy":                 {"ticker": "XLE",  "name": "Energy Select SPDR",   "market": "overseas", "exchange": "NYSE"},
    "materials":              {"ticker": "XLB",  "name": "Materials Select SPDR", "market": "overseas", "exchange": "NYSE"},
    "industrials":            {"ticker": "XLI",  "name": "Industrials Select SPDR","market": "overseas","exchange": "NYSE"},
    "consumer_discretionary": {"ticker": "XLY",  "name": "Consumer Discr SPDR",  "market": "overseas", "exchange": "NYSE"},
    "consumer_staples":       {"ticker": "XLP",  "name": "Consumer Staples SPDR", "market": "overseas", "exchange": "NYSE"},
    "healthcare":             {"ticker": "XLV",  "name": "Health Care SPDR",      "market": "overseas", "exchange": "NYSE"},
    "financials":             {"ticker": "XLF",  "name": "Financials SPDR",       "market": "overseas", "exchange": "NYSE"},
    "information_technology": {"ticker": "XLK",  "name": "Technology SPDR",       "market": "overseas", "exchange": "NYSE"},
    "communication_services": {"ticker": "XLC",  "name": "Communication SPDR",    "market": "overseas", "exchange": "NYSE"},
    "utilities":              {"ticker": "XLU",  "name": "Utilities SPDR",        "market": "overseas", "exchange": "NYSE"},
    "real_estate":            {"ticker": "XLRE", "name": "Real Estate SPDR",      "market": "overseas", "exchange": "NYSE"},
}

# 훈련 시 sectors.yaml 기준 섹터 순서 (sector_id 매핑 일치)
SECTOR_ORDER = [
    "energy", "materials", "industrials", "consumer_discretionary",
    "consumer_staples", "healthcare", "financials", "information_technology",
    "communication_services", "utilities", "real_estate",
]


def get_instrument(sector: str, execution_market: str = "kospi") -> dict:
    """섹터명으로 실제 투자 종목 정보 반환.

    Args:
        sector: 섹터명 (예: "energy")
        execution_market: "kospi" | "nasdaq"

    Returns:
        {"ticker": ..., "name": ..., "market": "domestic" | "overseas", ...}
    """
    if execution_market == "kospi":
        instrument = KOSPI_ETF_MAP.get(sector)
    else:
        instrument = NASDAQ_ETF_MAP.get(sector)

    if instrument is None:
        logger.warning(f"알 수 없는 섹터: {sector} → 스킵")
    return instrument


def get_all_instruments(execution_market: str = "kospi") -> dict[str, dict]:
    """전체 섹터 → 종목 매핑 반환."""
    etf_map = KOSPI_ETF_MAP if execution_market == "kospi" else NASDAQ_ETF_MAP
    return {sector: info for sector, info in etf_map.items()}


def get_sector_order() -> list[str]:
    """훈련 시 sector_id와 일치하는 섹터 순서 반환."""
    return SECTOR_ORDER.copy()
