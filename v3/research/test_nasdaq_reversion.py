"""NASDAQ reversion 1차 검증 (V4, survivor-only, 2026-05-29).

미국은 mean-reverting (V4: NASDAQ 중소형 momentum cs_ic -0.054 = reversion +0.054).
24시간 정책의 미국(밤) 축으로 reversion이 통하는지 1차 확인.

주의: survivor-only (FDR/yfinance 미국 상폐 데이터 없음). reversion은 survivor
bias가 신호를 **부풀림** (빠지고 상폐된 종목 누락 → 반등한 것만 보임). 따라서
1차 신호가 강해도 보수적 해석, 결제(EODHD $20) 후 survivorship-free 확정 필요.

측정:
  - cross-sectional reversion IC: past20 return rank vs fwd20 return rank (음수면 reversion)
  - reversion backtest: 각 rebalance(20d) past20 하위 N (많이 빠진) → long 20d
  - 여러 lookback(5/20/60d)

Usage:
    PYTHONPATH=. python v3/research/test_nasdaq_reversion.py
"""

from __future__ import annotations

import sys, json, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

# 미국 중소형 위주 + 대형 일부 (reversion은 중소형에서 강함). survivor-only.
UNIVERSE = [
    # 중소형 (V4 검증 리스트)
    "APPN", "BL", "PD", "AI", "BRZE", "ASAN", "FSLY", "GTLB", "BEAM", "NTLA",
    "RARE", "FOLD", "YETI", "SHAK", "WING", "CAKE", "PLAY", "DNUT", "UPST",
    "LMND", "AMBA", "POWI", "INDI", "DOCS", "PRVA", "OMCL", "PGNY", "SITM",
    "CEVA", "RMBS", "CRK", "SM", "MGY", "AR", "PLUG", "FUBO", "RIOT", "MARA",
    "SOFI", "OPEN", "DKNG", "RKLB", "IONQ", "SMCI", "ARM", "PLTR", "COIN",
    # 대형 (reversion 약할 것, 비교용)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
]
LB_LIST = [5, 20, 60]
HOLD = 20
REBAL_PER_YEAR = 252 / HOLD


def fetch(t, start, end):
    import yfinance as yf
    try:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 200:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception:
        return None


def main() -> int:
    logger.info(f"Fetching {len(UNIVERSE)} US tickers (survivor-only)...")
    series = {}
    for t in UNIVERSE:
        s = fetch(t, "2022-05-01", "2026-05-01")
        if s is not None:
            series[t] = s
    piv = pd.DataFrame(series).sort_index()
    logger.info(f"  fetched {piv.shape[1]} tickers, {piv.shape[0]} days")

    from scipy.stats import spearmanr
    dates = piv.index.tolist()

    logger.info("=" * 64)
    logger.info("NASDAQ REVERSION (survivor-only, cross-sectional IC + backtest)")
    logger.info("음수 momentum IC = reversion. backtest: past 하위 long")
    logger.info("=" * 64)

    out = {"momentum_ic_by_lb": {}, "reversion_backtest": {}}
    # 1. cross-sectional momentum IC (음수 = reversion)
    for lb in LB_LIST:
        ics = []
        for i in range(lb, len(dates) - HOLD, HOLD):
            past = (piv.iloc[i] / piv.iloc[i - lb] - 1.0)
            fwd = (piv.iloc[i + HOLD] / piv.iloc[i] - 1.0)
            v = past.notna() & fwd.notna()
            past, fwd = past[v], fwd[v]
            if len(past) < 8 or past.std() < 1e-9 or fwd.std() < 1e-9:
                continue
            rho, _ = spearmanr(past.to_numpy(), fwd.to_numpy())
            if np.isfinite(rho):
                ics.append(float(rho))
        mic = float(np.mean(ics)) if ics else 0.0
        out["momentum_ic_by_lb"][lb] = mic
        tag = " → REVERSION" if mic < -0.02 else (" → momentum" if mic > 0.02 else " → noise")
        logger.info(f"  lb{lb:2d}: momentum IC={mic:+.4f}{tag}  (reversion = {-mic:+.4f})")

    # 2. reversion backtest: 각 20d, past20 하위 N long (많이 빠진 것 매수)
    logger.info("")
    for n in [10, 20]:
        for lb in [5, 20]:
            rets = []
            for i in range(lb, len(dates) - HOLD, HOLD):
                past = (piv.iloc[i] / piv.iloc[i - lb] - 1.0)
                fwd = (piv.iloc[i + HOLD] / piv.iloc[i] - 1.0)
                v = past.notna() & fwd.notna()
                past, fwd = past[v], fwd[v]
                if len(past) < 10:
                    continue
                losers = past.nsmallest(min(n, len(past))).index  # 많이 빠진 것
                rets.append(float(fwd[losers].mean()) - 0.001)  # 미국 비용 0.1%
            rets = np.array(rets)
            if len(rets) == 0:
                continue
            eq = np.cumprod(1 + rets)
            ann = float(eq[-1] ** (REBAL_PER_YEAR / len(rets)) - 1)
            vol = float(rets.std() * np.sqrt(REBAL_PER_YEAR))
            sharpe = float(ann / vol) if vol > 1e-9 else 0.0
            peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
            out["reversion_backtest"][f"lb{lb}_n{n}"] = {
                "annual": ann, "sharpe": sharpe, "mdd": mdd, "n_rebal": len(rets)}
            logger.info(f"  reversion lb{lb} N={n}: annual={ann:+.1%}  Sharpe={sharpe:+.2f}  MDD={mdd:.1%}")

    logger.info("")
    logger.info("VERDICT (survivor-only, 부풀림 주의):")
    best_rev = max((abs(v) for v in out["momentum_ic_by_lb"].values()), default=0)
    if best_rev >= 0.04:
        logger.info(f"  신호 있음 (|IC|={best_rev:.4f}) → EODHD $20 결제로 survivorship-free 확정 권고")
        logger.info(f"    (survivor 부풀림 감안하면 실제는 더 작음)")
    else:
        logger.info(f"  신호 약함 (|IC|={best_rev:.4f}) → survivor에서도 약하면 결제 불필요")
    logger.info("=" * 64)

    p = Path("v3/research/reports/nasdaq_reversion.json")
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "n_tickers": piv.shape[1], **out}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
