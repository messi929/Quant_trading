"""V4 Korea 데이터 레이어 — survivorship-free PIT 패널.

backtest: 검증 캐시(parquet) 로드. live(Stage 2/3): build_panel 로 FDR 최신 수집.
패널 = (close pivot, dollar-volume pivot). 거래대금 = close × volume (PIT 유동성).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 검증된 survivorship-free 캐시 (korea_long_history.py 생성, 2014~, top600 live + 2015~상폐)
DEFAULT_CACHE = Path("v3/research/reports/korea_kosdaq_long_cache.parquet")


def load_panel(cache_path: Path = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """캐시 parquet → (close, dollar_volume) pivot. columns=ticker, index=date."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"패널 캐시 없음: {cache_path}. v3/research/korea_long_history.py --build 로 생성."
        )
    panel = pd.read_parquet(cache_path)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    return close, close * vol


def load_index(index_code: str, start: str, end: str) -> pd.Series:
    """지수 종가 series (regime gate 기준). FDR. live/backtest 공용."""
    import FinanceDataReader as fdr
    s = fdr.DataReader(index_code, start, end)["Close"]
    s.index = pd.to_datetime(s.index).normalize()
    return s
