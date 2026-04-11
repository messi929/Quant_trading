"""Ticker format normalization utilities."""


def kis_code(ticker: str) -> str:
    """KIS API용 6자리 코드 반환 — .KS/.KQ suffix 제거."""
    return ticker.split(".")[0]


def kospi_tick_size(price: float) -> int:
    """KOSPI/KOSDAQ 호가 단위(틱 사이즈) 반환."""
    if price < 2_000:
        return 1
    elif price < 5_000:
        return 5
    elif price < 20_000:
        return 10
    elif price < 50_000:
        return 50
    elif price < 200_000:
        return 100
    elif price < 500_000:
        return 500
    else:
        return 1_000


def round_to_tick(price: float, side: str = "buy") -> int:
    """가격을 호가 단위에 맞게 라운딩. 매수: 올림, 매도: 내림."""
    if price <= 0:
        return 0
    tick = kospi_tick_size(price)
    if side == "buy":
        import math
        return int(math.ceil(price / tick) * tick)
    else:
        return int(price // tick * tick)


def is_domestic(ticker: str) -> bool:
    """KOSPI/KOSDAQ 종목 여부 판별."""
    return kis_code(ticker).isdigit()


def market_of(ticker: str) -> str:
    """종목코드로 시장 판별."""
    if ticker.endswith(".KS"):
        return "KOSPI"
    elif ticker.endswith(".KQ"):
        return "KOSDAQ"
    elif kis_code(ticker).isdigit():
        return "KOSPI"
    else:
        return "NASDAQ"
