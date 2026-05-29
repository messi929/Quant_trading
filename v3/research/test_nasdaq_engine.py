"""나스닥 엔진 조건 탐색 (V4, survivor-only 윤곽, 2026-05-29).

나스닥 고유 조건 (한국 momentum 엔진과 독립):
  1. Long-short (market-neutral) — 과매도 long + 과매수 short. 미국 공매도 가능.
     한국 long-only와 달리 시장 방향 무관 순수 reversion alpha. MDD↓ 기대.
  2. 단기 horizon — 3/5일 (한국 60일과 다름). 1차에서 lb5 reversion 강함.
  3. VIX regime filter — 고변동성(과민반응)에서 reversion 강한지.

비교: long-only vs long-short, horizon별, VIX filter 유무.
목표: long-short market-neutral이 MDD를 크게 줄이는지 (미국 고유 장점).

주의: survivor-only (부풀림). 윤곽 탐색용 — 방향 잡히면 EODHD $20 확정.

Usage:
    PYTHONPATH=. python v3/research/test_nasdaq_engine.py
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

# 미국 중소형/중형 (long-short 위해 확대). survivor-only.
UNIVERSE = [
    "APPN","BL","PD","AI","BRZE","ASAN","FSLY","GTLB","BEAM","NTLA","RARE","FOLD",
    "YETI","SHAK","WING","CAKE","PLAY","DNUT","UPST","LMND","AMBA","POWI","INDI",
    "DOCS","PRVA","OMCL","PGNY","SITM","CEVA","RMBS","CRK","SM","MGY","AR","PLUG",
    "FUBO","RIOT","MARA","SOFI","OPEN","DKNG","RKLB","IONQ","SMCI","ARM","PLTR",
    "COIN","ROKU","DDOG","NET","SNOW","CRWD","ZS","OKTA","TWLO","DOCU","ZM","PINS",
    "SNAP","U","RBLX","HOOD","AFRM","BILL","S","PATH","DASH","ABNB","RDDT","HUBS",
    "TTD","MDB","NET","ESTC","TEAM","WDAY","ZI","GTLB","CFLT","FROG",
]
REBAL = 5
QUANTILE = 0.2   # 하위/상위 20%
COST = 0.001     # 미국 편도


def fetch(t, start, end):
    import yfinance as yf
    try:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 200:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        s = df["Close"].copy(); s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception:
        return None


def backtest(piv, vix, lb, mode, vix_filter=False, vix_pctl=0.5):
    """mode: 'long_only' (하위 long) or 'long_short' (하위 long - 상위 short)."""
    dates = piv.index.tolist()
    vix_thresh = vix.quantile(vix_pctl) if vix is not None else None
    rets = []
    for i in range(lb, len(dates) - REBAL, REBAL):
        d = dates[i]
        if vix_filter and vix is not None:
            vv = vix.asof(d)
            if pd.isna(vv) or vv < vix_thresh:   # 저변동성 → 진입 안 함 (현금)
                rets.append(0.0); continue
        past = (piv.iloc[i] / piv.iloc[i - lb] - 1.0)
        fwd = (piv.iloc[i + REBAL] / piv.iloc[i] - 1.0)
        v = past.notna() & fwd.notna()
        past, fwd = past[v], fwd[v]
        if len(past) < 15:
            rets.append(0.0); continue
        k = max(int(len(past) * QUANTILE), 3)
        losers = past.nsmallest(k).index    # 과매도 → long
        winners = past.nlargest(k).index    # 과매수 → short
        long_ret = float(fwd[losers].mean())
        if mode == "long_only":
            rets.append(long_ret - COST)
        else:  # long_short market-neutral
            short_ret = float(fwd[winners].mean())
            rets.append((long_ret - short_ret) - COST * 2)  # 양쪽 비용
    rets = np.array(rets)
    if len(rets) == 0:
        return {}
    per_year = 252 / REBAL
    eq = np.cumprod(1 + rets)
    ann = float(eq[-1] ** (per_year / len(rets)) - 1)
    vol = float(rets.std() * np.sqrt(per_year))
    sharpe = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sharpe, "mdd": mdd,
            "win": float((rets > 0).mean()), "n": len(rets)}


def main() -> int:
    logger.info(f"Fetching {len(set(UNIVERSE))} US tickers + VIX...")
    series = {}
    for t in set(UNIVERSE):
        s = fetch(t, "2022-05-01", "2026-05-01")
        if s is not None:
            series[t] = s
    piv = pd.DataFrame(series).sort_index()
    vix = fetch("^VIX", "2022-05-01", "2026-05-01")
    logger.info(f"  {piv.shape[1]} tickers, {piv.shape[0]} days, VIX={'ok' if vix is not None else 'fail'}")

    logger.info("=" * 70)
    logger.info("나스닥 엔진 조건 탐색 (survivor-only, 5d rebalance)")
    logger.info("=" * 70)
    out = {}
    for lb in [3, 5, 10]:
        lo = backtest(piv, vix, lb, "long_only")
        ls = backtest(piv, vix, lb, "long_short")
        lsv = backtest(piv, vix, lb, "long_short", vix_filter=True, vix_pctl=0.5)
        out[f"lb{lb}"] = {"long_only": lo, "long_short": ls, "long_short_vix": lsv}
        logger.info(f"--- lookback {lb}d ---")
        logger.info(f"  long_only      : annual={lo['annual']:+.1%}  Sharpe={lo['sharpe']:+.2f}  MDD={lo['mdd']:.1%}")
        logger.info(f"  long_short(MN) : annual={ls['annual']:+.1%}  Sharpe={ls['sharpe']:+.2f}  MDD={ls['mdd']:.1%}")
        logger.info(f"  long_short+VIX : annual={lsv['annual']:+.1%}  Sharpe={lsv['sharpe']:+.2f}  MDD={lsv['mdd']:.1%}")
        logger.info("")

    logger.info("VERDICT (survivor-only 윤곽):")
    # long-short가 long-only보다 MDD 작은지 (market-neutral 효과)
    lb5 = out["lb5"]
    if lb5["long_short"]["sharpe"] > lb5["long_only"]["sharpe"] and abs(lb5["long_short"]["mdd"]) < abs(lb5["long_only"]["mdd"]):
        logger.info(f"  long-short market-neutral이 MDD↓ Sharpe↑ — 미국 고유 장점 확인.")
    logger.info(f"  best Sharpe 조합 → EODHD $20로 survivorship-free 확정 대상")
    logger.info("=" * 70)

    p = Path("v3/research/reports/nasdaq_engine.json")
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                            "n_tickers": piv.shape[1], "rebal": REBAL, "grid": out},
                           indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
