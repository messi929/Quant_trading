"""Ticker format normalization utilities.

KIS API는 6자리 숫자 코드만 허용 (예: "005930").
yfinance/parquet은 ".KS"/".KQ" suffix 포함 (예: "005930.KS").

이 모듈의 두 함수를 시스템 전체에서 사용하여 형식 불일치 버그 재발을 방지한다.
"""


def kis_code(ticker: str) -> str:
    """KIS API용 6자리 코드 반환 — .KS/.KQ suffix 제거.

    Examples:
        "005930.KS" → "005930"
        "039490.KQ" → "039490"
        "005930"    → "005930"   (suffix 없으면 그대로)
        "INTU"      → "INTU"    (미국 종목 그대로)
    """
    return ticker.split(".")[0]


def kospi_tick_size(price: float) -> int:
    """KOSPI/KOSDAQ 호가 단위(틱 사이즈) 반환.

    KRX 규정 기준 (2023~ 현행):
        가격 < 2,000       → 1원
        가격 < 5,000       → 5원
        가격 < 20,000      → 10원
        가격 < 50,000      → 50원
        가격 < 200,000     → 100원
        가격 < 500,000     → 500원
        가격 ≥ 500,000     → 1,000원
    """
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
    """가격을 호가 단위에 맞게 라운딩.

    매수: 올림 (체결 확률 ↑)
    매도: 내림 (체결 확률 ↑)

    Examples:
        round_to_tick(5274, "sell") → 5270  (10원 단위 내림)
        round_to_tick(8364, "sell") → 8360  (10원 단위 내림)
        round_to_tick(3021, "buy")  → 3025  (5원 단위 올림)
    """
    if price <= 0:
        return 0
    tick = kospi_tick_size(price)
    if side == "buy":
        # 올림: 이미 틱에 맞으면 그대로
        import math
        return int(math.ceil(price / tick) * tick)
    else:
        # 내림
        return int(price // tick * tick)


def is_domestic(ticker: str) -> bool:
    """KOSPI/KOSDAQ 종목 여부 판별.

    KIS 6자리 숫자 코드이면 국내, 그 외(알파벳)는 해외.
    ".KS"/".KQ" suffix가 있어도 정상 처리.

    Examples:
        "005930"    → True
        "005930.KS" → True
        "039490.KQ" → True
        "INTU"      → False
        "AAPL"      → False
    """
    return kis_code(ticker).isdigit()
