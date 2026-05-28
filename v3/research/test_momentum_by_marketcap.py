"""시총 그룹별 momentum 검증 (V4 도메인 — 중소형 가설, 2026-05-29).

가설: momentum/비효율은 중소형주에 있다. NASDAQ-100 초대형은 efficient라
trend IC −0.002였으나, NASDAQ 중소형·KOSDAQ는 momentum이 살아날 것.

측정 (그룹별):
  1. Time-series momentum (추세 추종의 진짜 신호):
     past_ret(lookback) 부호로 long → forward_ret. trend-following 모방.
     momentum premium = mean(fwd | past>0) − mean(fwd | past<0)
  2. Cross-sectional momentum IC: past_ret rank vs fwd_ret rank (Spearman)

그룹: NASDAQ 초대형 / NASDAQ 중소형 / KOSPI / KOSDAQ
한국 비용(거래세 ~0.3%)은 별도 차감 시나리오로 표시.

데이터: yfinance (.KS=KOSPI, .KQ=KOSDAQ). 안 받아지는 종목 skip.

Usage:
    PYTHONPATH=. python v3/research/test_momentum_by_marketcap.py
"""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

# ── 시총 그룹별 대표 샘플 (yfinance suffix: .KS=KOSPI, .KQ=KOSDAQ) ──
GROUPS = {
    "NASDAQ_megacap": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST",
        "NFLX", "AMD", "PEP", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "AMAT",
        "MU", "INTU", "ISRG", "BKNG", "ADI", "REGN", "VRTX",
    ],
    "NASDAQ_smallcap": [
        "APPN", "BL", "PD", "AI", "BRZE", "ASAN", "FSLY", "GTLB", "BEAM", "NTLA",
        "RARE", "FOLD", "YETI", "SHAK", "WING", "CAKE", "PLAY", "DNUT", "UPST",
        "LMND", "AMBA", "POWI", "INDI", "CRK", "DOCS", "PRVA", "OMCL", "PGNY",
        "SITM", "CEVA",
    ],
    "KOSPI": [
        "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
        "000270.KS", "005490.KS", "035420.KS", "035720.KS", "051910.KS",
        "006400.KS", "028260.KS", "105560.KS", "055550.KS", "012330.KS",
        "066570.KS", "003670.KS", "015760.KS", "017670.KS", "034730.KS",
        "009150.KS", "011200.KS", "086790.KS", "316140.KS", "024110.KS",
    ],
    "KOSDAQ": [
        "086520.KQ", "196170.KQ", "247540.KQ", "028300.KQ", "066970.KQ",
        "357780.KQ", "058470.KQ", "240810.KQ", "098460.KQ", "022100.KQ",
        "263750.KQ", "293490.KQ", "095340.KQ", "041510.KQ", "067310.KQ",
        "078600.KQ", "086900.KQ", "214150.KQ", "112040.KQ", "039030.KQ",
        "145020.KQ", "278280.KQ", "036930.KQ", "140860.KQ", "200670.KQ",
    ],
}

LOOKBACKS = [20, 60, 120]   # past return windows (trend lookback)
HORIZON = 20                # forward holding (trend-following은 길게)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 200:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()[["Date", "Close"]].copy()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = ticker
        return df
    except Exception:
        return None


def measure_group(panel: pd.DataFrame, lookback: int, horizon: int) -> dict:
    """Time-series momentum + cross-sectional momentum."""
    from scipy.stats import spearmanr
    piv = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dates = piv.index.tolist()

    ts_up, ts_down = [], []   # forward returns conditioned on past sign
    cs_ics = []               # cross-sectional momentum IC per date

    step = horizon
    for i in range(lookback, len(dates) - horizon, step):
        d = dates[i]
        past = (piv.iloc[i] / piv.iloc[i - lookback] - 1.0)
        fwd = (piv.iloc[i + horizon] / piv.iloc[i] - 1.0)
        valid = past.notna() & fwd.notna()
        past, fwd = past[valid], fwd[valid]
        if len(past) < 5:
            continue
        # TS momentum: 부호별 forward
        ts_up.extend(fwd[past > 0].tolist())
        ts_down.extend(fwd[past < 0].tolist())
        # CS momentum IC
        if past.std() > 1e-9 and fwd.std() > 1e-9:
            rho, _ = spearmanr(past.to_numpy(), fwd.to_numpy())
            if np.isfinite(rho):
                cs_ics.append(float(rho))

    ts_up_mean = float(np.mean(ts_up)) if ts_up else 0.0
    ts_down_mean = float(np.mean(ts_down)) if ts_down else 0.0
    return {
        "ts_momentum_premium": ts_up_mean - ts_down_mean,   # >0이면 추세추종 작동
        "ts_long_fwd_mean": ts_up_mean,                      # past>0 종목 forward (long-only 수익)
        "ts_down_fwd_mean": ts_down_mean,
        "cs_momentum_ic": float(np.mean(cs_ics)) if cs_ics else 0.0,
        "n_long": len(ts_up), "n_short": len(ts_down),
    }


def main() -> int:
    start, end = "2022-05-01", "2026-05-01"
    logger.info("Fetching market-cap group samples...")
    group_panels = {}
    for gname, tickers in GROUPS.items():
        frames, kept = [], []
        for t in tickers:
            df = fetch(t, start, end)
            if df is not None:
                frames.append(df)
                kept.append(t)
        if frames:
            group_panels[gname] = pd.concat(frames, ignore_index=True)
            logger.info(f"  {gname}: {len(kept)}/{len(tickers)} tickers")
        else:
            logger.warning(f"  {gname}: no data")

    logger.info("=" * 70)
    logger.info("MOMENTUM by market-cap group (HORIZON=20d hold)")
    logger.info("=" * 70)
    logger.info("ts_premium = mean(fwd|past>0) − mean(fwd|past<0) [추세추종 작동 여부]")
    logger.info("ts_long    = past>0 종목 평균 forward (long-only 추세추종 raw 수익)")
    logger.info("cs_ic      = cross-sectional momentum rank IC")
    logger.info("한국 비용 ~0.3%/회전 (20d hold면 월 1.5회전 ≈ 0.45%/월) 별도 차감 고려")
    logger.info("")

    report = {}
    for gname, panel in group_panels.items():
        logger.info(f"### {gname} ###")
        gres = {}
        for lb in LOOKBACKS:
            m = measure_group(panel, lb, HORIZON)
            gres[f"lookback_{lb}"] = m
            logger.info(
                f"  lookback {lb:3d}d: ts_premium={m['ts_momentum_premium']:+.4f}  "
                f"ts_long={m['ts_long_fwd_mean']:+.4f}  cs_ic={m['cs_momentum_ic']:+.4f}  "
                f"(n_long={m['n_long']})"
            )
        report[gname] = gres
        logger.info("")

    out = Path("v3/research/reports/momentum_by_marketcap.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon": HORIZON, "lookbacks": LOOKBACKS,
        "groups": report,
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
