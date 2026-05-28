"""한국 survivorship-FREE momentum 검증 (V4, 2026-05-29).

이전 간접 검증(sub-period)은 WARN(최근 편중). 이제 FinanceDataReader로 상폐
종목 OHLCV까지 받아 진짜 survivorship-free 검증.

핵심: 상폐 종목 OHLCV를 universe에 합치면 상폐 전 데이터는 포함되고 상폐 후는
NaN → 그 자체로 point-in-time. 폭락 후 상폐된 종목(추세추종이 손실 본 케이스)이
포함되므로, momentum이 살아남으면 진짜 edge, 무너지면 생존자 편향.

비교:
  - survivor-only: 현재 상장 종목만 (이전 결과 cs_ic +0.048)
  - survivorship-free: 현재 상장 + 2021~ 상폐 종목

데이터: FDR (무료). KOSDAQ 시총 상위 + 상폐 종목.

Usage:
    PYTHONPATH=. python v3/research/test_korea_survivorship_free.py
"""

from __future__ import annotations

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr

LOOKBACK = 60
HORIZON = 20
N_LIVE = 150        # 현재 상장 시총 상위
START = "2021-01-01"
END = "2026-05-01"


def fetch_ohlcv(code: str) -> pd.Series | None:
    try:
        df = fdr.DataReader(code, START, END)
        if df is None or df.empty or "Close" not in df.columns or len(df) < 80:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception:
        return None


def momentum(close_piv: pd.DataFrame, lookback: int, horizon: int) -> dict:
    from scipy.stats import spearmanr
    dates = close_piv.index.tolist()
    ts_up, ts_down, cs = [], [], []
    for i in range(lookback, len(dates) - horizon, horizon):
        past = (close_piv.iloc[i] / close_piv.iloc[i - lookback] - 1.0)
        fwd = (close_piv.iloc[i + horizon] / close_piv.iloc[i] - 1.0)
        v = past.notna() & fwd.notna()
        past, fwd = past[v], fwd[v]
        if len(past) < 8:
            continue
        ts_up.extend(fwd[past > 0].tolist())
        ts_down.extend(fwd[past < 0].tolist())
        if past.std() > 1e-9 and fwd.std() > 1e-9:
            rho, _ = spearmanr(past.to_numpy(), fwd.to_numpy())
            if np.isfinite(rho):
                cs.append(float(rho))
    return {
        "ts_premium": (np.mean(ts_up) - np.mean(ts_down)) if ts_up and ts_down else 0.0,
        "ts_long": float(np.mean(ts_up)) if ts_up else 0.0,
        "cs_ic": float(np.mean(cs)) if cs else 0.0,
        "n_long": len(ts_up),
    }


def main() -> int:
    # ── 1. universe 구성 ──
    logger.info("Building KOSDAQ universe (live top + delisted)...")
    kq = fdr.StockListing("KOSDAQ")
    kq["Marcap"] = pd.to_numeric(kq["Marcap"], errors="coerce")
    live = kq.nlargest(N_LIVE, "Marcap")["Code"].tolist()
    logger.info(f"  live (top {N_LIVE} by mktcap): {len(live)}")

    delist = fdr.StockListing("KRX-DELISTING")
    delist["DelistingDate"] = pd.to_datetime(delist["DelistingDate"], errors="coerce")
    kq_del = delist[
        (delist["Market"] == "KOSDAQ") & (delist["DelistingDate"] >= "2021-01-01")
    ]["Symbol"].tolist()
    # 정규 6자리 코드만
    kq_del = [c for c in kq_del if isinstance(c, str) and c.isdigit() and len(c) == 6]
    logger.info(f"  delisted 2021~ (KOSDAQ, valid code): {len(kq_del)}")

    # ── 2. OHLCV 수집 ──
    logger.info("Fetching OHLCV (FDR)...")
    live_series, del_series = {}, {}
    for i, c in enumerate(live, 1):
        s = fetch_ohlcv(c)
        if s is not None:
            live_series[c] = s
        if i % 50 == 0:
            logger.info(f"  live {i}/{len(live)}")
    logger.info(f"  live fetched: {len(live_series)}")
    for i, c in enumerate(kq_del, 1):
        s = fetch_ohlcv(c)
        if s is not None:
            del_series[c] = s
        if i % 50 == 0:
            logger.info(f"  delisted {i}/{len(kq_del)}")
    logger.info(f"  delisted fetched: {len(del_series)}")

    # ── 3. pivot + momentum ──
    live_piv = pd.DataFrame(live_series).sort_index()
    allp = dict(live_series)
    allp.update(del_series)
    full_piv = pd.DataFrame(allp).sort_index()

    logger.info("=" * 64)
    logger.info("SURVIVORSHIP-FREE vs SURVIVOR-ONLY (KOSDAQ, lb60 hold20)")
    logger.info("=" * 64)
    surv = momentum(live_piv, LOOKBACK, HORIZON)
    free = momentum(full_piv, LOOKBACK, HORIZON)
    logger.info(f"  survivor-only ({len(live_series)} live):")
    logger.info(f"    ts_premium={surv['ts_premium']:+.4f}  ts_long={surv['ts_long']:+.4f}  cs_ic={surv['cs_ic']:+.4f}")
    logger.info(f"  survivorship-FREE ({len(allp)} = live+delisted):")
    logger.info(f"    ts_premium={free['ts_premium']:+.4f}  ts_long={free['ts_long']:+.4f}  cs_ic={free['cs_ic']:+.4f}")
    logger.info("")
    # 감소율
    def shrink(a, b):
        return (b - a) / abs(a) * 100 if abs(a) > 1e-9 else 0.0
    logger.info(f"  cs_ic 변화: {surv['cs_ic']:+.4f} → {free['cs_ic']:+.4f} ({shrink(surv['cs_ic'], free['cs_ic']):+.0f}%)")
    logger.info(f"  ts_long 변화: {surv['ts_long']:+.4f} → {free['ts_long']:+.4f} ({shrink(surv['ts_long'], free['ts_long']):+.0f}%)")
    logger.info("")
    logger.info("VERDICT:")
    if free["cs_ic"] > 0.02 and free["ts_long"] > 0:
        logger.info("  PASS — 상폐 포함해도 momentum 양수 유지. 한국 momentum 진짜 edge.")
    elif free["cs_ic"] > 0:
        logger.info("  MARGINAL — 상폐 포함 시 약해지나 양수. 부분적 edge.")
    else:
        logger.info("  FAIL — 상폐 포함 시 momentum 소멸. 이전 신호는 생존자 편향.")
    logger.info("=" * 64)

    out = Path("v3/research/reports/korea_survivorship_free.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_live": len(live_series), "n_delisted": len(del_series),
        "lookback": LOOKBACK, "horizon": HORIZON,
        "survivor_only": surv, "survivorship_free": free,
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
