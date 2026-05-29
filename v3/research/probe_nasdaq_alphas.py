"""NASDAQ 다른 엣지 1회 probe — survivorship-free (V4, 2026-05-29).

reversion 기각(backtest_nasdaq_pit.py) 후, 같은 survivorship-free 패널
(reports/nasdaq_pit_cache.parquet, 6753 NASDAQ common, 상폐포함)로 다른 엣지 탐색.

가설: momentum은 reversion보다 survivorship bias에 강하다.
  reversion = 패자(상폐위험군) 매수 → 상폐 함정 직격. momentum = 승자 매수 →
  상폐 위험 낮은 종목 선택 → bias 영향 작음. + 한국에서 검증된 메커니즘.

테스트 (전부 long-only, hold 20d, PIT 거래대금 universe top30 mega 제외):
  1. TS momentum   : past LB > 0 인 것 중 momentum 상위 N
  2. CS momentum   : 부호 무관 momentum 상위 N
  3. low-vol       : 20d 실현변동성 하위 N (저변동 anomaly)
각 lb {20,60,120}. 상폐 손실(PEN) 동일 반영. raw + VIX-filter 비교.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/probe_nasdaq_alphas.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

REPORTS = Path("v3/research/reports")
CACHE = REPORTS / "nasdaq_pit_cache.parquet"
START, END = "2021-06-01", "2026-05-01"

HOLD = 20
N_POS = 20
SKIP_TOP = 30
POOL = 300
MIN_PRICE = 5.0
PEN = 0.30
COST = 0.001
RPY = 252 / HOLD


def fwd_ret(adj, dates, i, picks):
    pr = []
    for t in picks:
        e = adj.iloc[i][t]
        x = adj.iloc[i + HOLD][t]
        if pd.notna(x):
            pr.append(min(x / e - 1.0, 5.0))
        else:
            w = adj.iloc[i:i + HOLD + 1][t]
            lv = w.last_valid_index()
            pr.append((w[lv] / e - 1.0 - PEN) if lv is not None and w[lv] > 0 else -1.0)
    return pr


def stats(rets):
    r = np.array(rets)
    if len(r) == 0 or not np.all(np.isfinite(r)):
        return {}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "win": float((r > 0).mean()), "n": len(r)}


def backtest(adj, liq20, vol20, dates, lb, mode, vix=None, vix_filter=False):
    vthr = vix.quantile(0.5) if vix is not None else None
    rets, prev = [], set()
    for i in range(lb, len(dates) - HOLD, HOLD):
        if vix_filter and vix is not None:
            vv = vix.asof(dates[i])
            if pd.isna(vv) or vv < vthr:
                rets.append(0.0); prev = set(); continue
        liq = liq20.iloc[i].dropna()
        if len(liq) < SKIP_TOP + 30:
            rets.append(0.0); prev = set(); continue
        pool = liq.sort_values(ascending=False).iloc[SKIP_TOP:SKIP_TOP + POOL].index
        pi, pl = adj.iloc[i][pool], adj.iloc[i - lb][pool]
        valid = np.isfinite(pi / pl) & (pi >= MIN_PRICE) & (pl > 0)
        pool = pi[valid].index
        if len(pool) < 20:
            rets.append(0.0); prev = set(); continue
        if mode in ("ts_mom", "cs_mom"):
            mom = (adj.iloc[i][pool] / adj.iloc[i - lb][pool] - 1.0).dropna()
            if mode == "ts_mom":
                mom = mom[mom > 0]
            if len(mom) == 0:
                rets.append(0.0); prev = set(); continue
            picks = mom.nlargest(min(N_POS, len(mom))).index
        else:  # low_vol
            vv = vol20.iloc[i][pool].dropna()
            vv = vv[vv > 0]
            if len(vv) < N_POS:
                rets.append(0.0); prev = set(); continue
            picks = vv.nsmallest(N_POS).index
        pr = fwd_ret(adj, dates, i, picks)
        gross = float(np.mean(pr))
        to = len(set(picks) - prev) / max(len(picks), 1)
        rets.append(gross - COST * to); prev = set(picks)
    return stats(rets)


def fetch_vix():
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=START, end=END, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        s = df["Close"].copy(); s.index = pd.to_datetime(s.index).normalize(); return s
    except Exception:
        return None


def main() -> int:
    panel = pd.read_parquet(CACHE)
    adj = panel.pivot_table(index="date", columns="ticker", values="adj").sort_index()
    dvol = panel.pivot_table(index="date", columns="ticker", values="dollarvol").sort_index()
    dates = adj.index.tolist()
    liq20 = dvol.rolling(20, min_periods=10).mean()
    vol20 = adj.pct_change().rolling(20, min_periods=10).std()
    vix = fetch_vix()
    logger.info(f"pool {adj.shape[1]} tickers, {adj.shape[0]} days, VIX={'ok' if vix is not None else 'fail'}")

    logger.info("=" * 74)
    logger.info(f"NASDAQ 다른 엣지 probe — survivorship-free (long-only, hold {HOLD}d, N={N_POS})")
    logger.info("=" * 74)
    out = {}
    for mode in ["ts_mom", "cs_mom", "low_vol"]:
        logger.info(f"--- {mode} ---")
        for lb in [20, 60, 120]:
            raw = backtest(adj, liq20, vol20, dates, lb, mode)
            vf = backtest(adj, liq20, vol20, dates, lb, mode, vix=vix, vix_filter=True)
            out[f"{mode}_lb{lb}"] = {"raw": raw, "vix": vf}
            logger.info(f"  lb{lb:<3d} raw : annual={raw.get('annual',0):+.1%} Sharpe={raw.get('sharpe',0):+.2f} "
                        f"MDD={raw.get('mdd',0):.1%} win={raw.get('win',0):.0%}  |  "
                        f"VIX: annual={vf.get('annual',0):+.1%} Sharpe={vf.get('sharpe',0):+.2f} MDD={vf.get('mdd',0):.1%}")
        logger.info("")

    flat = [(k, v["raw"]) for k, v in out.items()] + [(k + "+vix", v["vix"]) for k, v in out.items()]
    best = max(flat, key=lambda kv: kv[1].get("sharpe", -9))
    logger.info("VERDICT (survivorship-free):")
    logger.info(f"  best: {best[0]}  Sharpe={best[1].get('sharpe',0):.2f} annual={best[1].get('annual',0):+.1%} MDD={best[1].get('mdd',0):.1%}")
    logger.info(f"  (reversion 기각: best Sharpe -0.47. 한국 momentum: KOSDAQ +20.5%/0.61)")
    logger.info("=" * 74)

    p = REPORTS / "nasdaq_alpha_probe.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "pool_tickers": int(adj.shape[1]), "hold": HOLD, "n_pos": N_POS,
                             "skip_top": SKIP_TOP, "pool": POOL, "min_price": MIN_PRICE,
                             "results": out}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
