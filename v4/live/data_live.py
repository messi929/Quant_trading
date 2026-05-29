"""V4 live 데이터 — 현재 KOSDAQ universe + 최근 OHLCV → 엔진 입력 패널.

live 는 미래 상폐 걱정 없음 → universe = 현재 시총 상위 N (PIT 거래대금으로 다시 선별).
lookback_days 만큼만 (max_lb=120 거래일 + 버퍼). backtest 의 긴 캐시와 달리 짧음.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from v4.config import KoreaConfig


def build_live_panel(cfg: KoreaConfig = KoreaConfig(), universe_size: int = 400,
                     lookback_days: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(close, dollar_volume, index) — 오늘까지. FDR. 느림(N stock 순차 ~수분)."""
    import FinanceDataReader as fdr

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    listing = fdr.StockListing(cfg.market)
    listing["Marcap"] = pd.to_numeric(listing["Marcap"], errors="coerce")
    codes = listing.nlargest(universe_size, "Marcap")["Code"].tolist()
    logger.info(f"live universe: {cfg.market} 시총 상위 {len(codes)}, OHLCV {start}~{end} 수집...")

    frames = []
    for j, c in enumerate(codes, 1):
        try:
            df = fdr.DataReader(c, start, end)
            if df is None or df.empty or len(df) < cfg.max_lb:
                continue
            s = df[["Close", "Volume"]].copy()
            s.columns = ["close", "volume"]
            s["date"] = pd.to_datetime(s.index).normalize()
            s["ticker"] = c
            frames.append(s.reset_index(drop=True))
        except Exception:
            pass
        if j % 100 == 0:
            logger.info(f"  {j}/{len(codes)} (ok={len(frames)})")

    panel = pd.concat(frames, ignore_index=True)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    index = fdr.DataReader(cfg.index_code, start, end)["Close"]
    index.index = pd.to_datetime(index.index).normalize()
    logger.info(f"패널: {close.shape[1]} ticker, {close.shape[0]} 거래일 (~{close.index[-1].date()})")
    return close, close * vol, index
