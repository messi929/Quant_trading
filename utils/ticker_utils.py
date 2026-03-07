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
