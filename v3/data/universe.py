"""Ticker universe management — KOSPI200 + NASDAQ-100."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass
class TickerInfo:
    ticker: str        # yfinance 형식: "005930.KS", "AAPL"
    name: str
    market: str        # "KOSPI", "KOSDAQ", "NASDAQ"


class Universe:
    """Manages the trading universe of tickers across markets."""

    def __init__(self, markets: list[str] | None = None, kospi200_only: bool = True):
        self.markets = markets or ["KOSPI", "NASDAQ"]
        self.kospi200_only = kospi200_only
        self._tickers: list[TickerInfo] = []

    def build(self) -> list[TickerInfo]:
        """Build full ticker universe from configured markets."""
        self._tickers = []

        if "KOSPI" in self.markets:
            self._tickers.extend(self._get_kospi_tickers())
        if "KOSDAQ" in self.markets:
            self._tickers.extend(self._get_kosdaq_tickers())
        if "NASDAQ" in self.markets:
            self._tickers.extend(self._get_nasdaq_tickers())

        logger.info(f"Universe built: {len(self._tickers)} tickers "
                     f"({', '.join(f'{m}: {self.count(m)}' for m in self.markets)})")
        return self._tickers

    def count(self, market: str | None = None) -> int:
        if market is None:
            return len(self._tickers)
        return sum(1 for t in self._tickers if t.market == market)

    @property
    def tickers(self) -> list[TickerInfo]:
        return self._tickers

    def ticker_list(self, market: str | None = None) -> list[str]:
        if market is None:
            return [t.ticker for t in self._tickers]
        return [t.ticker for t in self._tickers if t.market == market]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"ticker": t.ticker, "name": t.name, "market": t.market}
            for t in self._tickers
        ])

    # KOSPI 대형주 (시가총액 상위, 유동성 확보)
    KOSPI_LARGE_CAP = [
        ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("373220", "LG에너지솔루션"),
        ("207940", "삼성바이오로직스"), ("005380", "현대자동차"), ("000270", "기아"),
        ("006400", "삼성SDI"), ("051910", "LG화학"), ("035420", "NAVER"),
        ("005490", "POSCO홀딩스"), ("035720", "카카오"), ("028260", "삼성물산"),
        ("105560", "KB금융"), ("055550", "신한지주"), ("012330", "현대모비스"),
        ("066570", "LG전자"), ("003670", "포스코퓨처엠"), ("086790", "하나금융지주"),
        ("032830", "삼성생명"), ("034730", "SK"), ("096770", "SK이노베이션"),
        ("003550", "LG"), ("015760", "한국전력"), ("033780", "KT&G"),
        ("000810", "삼성화재"), ("018260", "삼성에스디에스"), ("034020", "두산에너빌리티"),
        ("017670", "SK텔레콤"), ("009150", "삼성전기"), ("024110", "HMM"),
        ("010130", "고려아연"), ("010950", "S-Oil"), ("011200", "HJ중공업"),
        ("316140", "우리금융지주"), ("259960", "크래프톤"), ("030200", "KT"),
        ("047050", "포스코인터내셔널"), ("036570", "엔씨소프트"), ("011170", "롯데케미칼"),
        ("138040", "메리츠금융지주"), ("003490", "대한항공"), ("352820", "하이브"),
        ("068270", "셀트리온"), ("009540", "한국조선해양"), ("010140", "삼성중공업"),
        ("267250", "HD현대"), ("042660", "한화오션"), ("009830", "한화솔루션"),
        ("006360", "GS건설"), ("000720", "현대건설"),
    ]

    def _get_kospi_tickers(self) -> list[TickerInfo]:
        """KOSPI 대형주 목록 (하드코딩, pykrx 로컬 이슈 우회)."""
        # Try pykrx first
        try:
            from pykrx import stock as pykrx_stock
            from datetime import datetime, timedelta

            date_str = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
            codes = pykrx_stock.get_market_ticker_list(date_str, market="KOSPI")
            if hasattr(codes, 'tolist'):
                codes = codes.tolist()
            codes = list(codes) if codes else []

            if len(codes) > 50:
                result = []
                for code in codes[:200]:
                    try:
                        name = pykrx_stock.get_market_ticker_name(code)
                        result.append(TickerInfo(ticker=f"{code}.KS", name=name, market="KOSPI"))
                    except Exception:
                        continue
                if result:
                    logger.info(f"KOSPI tickers (pykrx): {len(result)}")
                    return result
        except Exception:
            pass

        # Fallback: hardcoded large-cap list
        result = [
            TickerInfo(ticker=f"{code}.KS", name=name, market="KOSPI")
            for code, name in self.KOSPI_LARGE_CAP
        ]
        logger.info(f"KOSPI tickers (hardcoded): {len(result)}")
        return result

    def _get_kosdaq_tickers(self) -> list[TickerInfo]:
        """KOSDAQ 대형주 목록."""
        # pykrx 로컬 이슈로 하드코딩 fallback
        KOSDAQ_LARGE = [
            ("247540", "에코프로비엠"), ("086520", "에코프로"),
            ("403870", "HPSP"), ("328130", "루닛"), ("041510", "에스엠"),
            ("145020", "휴젤"), ("293490", "카카오게임즈"), ("263750", "펄어비스"),
            ("112040", "위메이드"), ("039030", "이오테크닉스"),
        ]
        result = [
            TickerInfo(ticker=f"{code}.KQ", name=name, market="KOSDAQ")
            for code, name in KOSDAQ_LARGE
        ]
        logger.info(f"KOSDAQ tickers: {len(result)}")
        return result

    def _get_nasdaq_tickers(self) -> list[TickerInfo]:
        """NASDAQ-100 구성 종목 (S&P 500과의 교집합)."""
        try:
            # 하드코딩된 상위 종목 목록 (안정적, Wikipedia 스크래핑 불필요)
            nasdaq100_core = [
                "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA",
                "AVGO", "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS",
                "INTC", "INTU", "CMCSA", "TXN", "QCOM", "AMGN", "ISRG", "AMAT",
                "HON", "BKNG", "LRCX", "VRTX", "MU", "ADI", "REGN", "KLAC",
                "PANW", "ADP", "SNPS", "CDNS", "MDLZ", "SBUX", "MELI", "GILD",
                "PYPL", "ASML", "CTAS", "CRWD", "CSX", "MAR", "ORLY", "MNST",
                "ABNB", "MRVL", "NXPI", "FTNT", "DASH", "PCAR", "WDAY", "CEG",
                "ODFL", "ROST", "AEP", "CPRT", "FAST", "DXCM", "PAYX", "KDP",
                "EXC", "CTSH", "MRNA", "GEHC", "KHC", "VRSK", "EA", "XEL",
                "LULU", "CCEP", "IDXX", "MCHP", "TTWO", "CSGP", "FANG", "ANSS",
                "ON", "ZS", "DDOG", "CDW", "GFS", "BIIB", "TEAM", "ILMN",
                "WBD", "DLTR", "MDB", "ARM", "BKR", "TTD", "SMCI", "PDD",
                "LIN", "AZN", "COIN", "PLTR",
            ]

            result = [
                TickerInfo(ticker=t, name=t, market="NASDAQ")
                for t in nasdaq100_core
            ]
            logger.info(f"NASDAQ tickers: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Failed to get NASDAQ tickers: {e}")
            return []
