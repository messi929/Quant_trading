"""KOSPI 엣지 탐색 — 가격 기반 (V4, 2026-05-30).

KOSPI momentum full-cycle Sharpe 0.15 (효율적 대형주, NASDAQ-100과 동일 이유). KOSPI에
맞는 다른 엣지가 있는지: reversion(대형주 평균회귀) / low-vol(저변동 이상현상) /
momentum 변형. 전부 가격 기반 (펀더멘털 불요). 배포가능(Sharpe~0.5+) 나오면 설계.

universe: KOSPI long 캐시(2014~, survivorship-free) PIT 거래대금 top100. long-only,
equal-weight N=20, 상폐손실+비용 0.4% 동일. regime=KS11 200d SMA.

raw(gate 없음) 먼저 base 엣지 확인 → best 에 gate+vol-target overlay.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_kospi_edge.py
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

import FinanceDataReader as fdr

LIQ_TOP, N_POS = 100, 20
COST, DELIST_PEN, MIN_PRICE = 0.004, 0.5, 1000.0
TARGET_VOL, VOL_WIN, CAP = 0.15, 6, 1.5
CACHE = Path("v3/research/reports/korea_kospi_long_cache.parquet")


def pos_ret(path, entry):
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    return (path[lv] / entry - 1.0 - DELIST_PEN) if lv is not None and path[lv] > 0 else -1.0


def picks_at(close, dvol, vol20, i, lb, kind):
    """kind: reversion(과매도 하위20%) / lowvol(저변동) / momentum(추세 상위)."""
    liq = dvol.iloc[i].dropna()
    if len(liq) < 20:
        return []
    pool = liq.nlargest(LIQ_TOP).index
    pi = close.iloc[i][pool]
    pool = pi[(pi >= MIN_PRICE)].index
    if len(pool) < 20:
        return []
    if kind == "lowvol":
        vv = vol20.iloc[i][pool].dropna(); vv = vv[vv > 0]
        return list(vv.nsmallest(N_POS).index) if len(vv) >= N_POS else []
    past = (close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0).dropna()
    if kind == "reversion":
        k = max(int(len(past) * 0.2), 3)
        return list(past.nsmallest(k).index)[:N_POS]      # 과매도 → long
    # momentum
    trend = past[past > 0]
    return list(trend.nlargest(min(N_POS, len(trend))).index) if len(trend) else []


def basket(close, dvol, vol20, lb, hold, kind):
    dates = close.index.tolist(); out = {}
    for i in range(max(lb, 20), len(dates) - hold, hold):
        picks = picks_at(close, dvol, vol20, i, lb, kind)
        if not picks:
            out[dates[i]] = 0.0; continue
        pr = [pos_ret(close.iloc[i:i + hold + 1][t], close.iloc[i][t]) for t in picks]
        out[dates[i]] = float(np.mean(pr)) - COST
    return pd.Series(out).sort_index()


def overlay(b, gate, hold, use_vt):
    rpy = 252 / hold; tgt = TARGET_VOL / np.sqrt(rpy); vals = b.values; out = []
    for k, (d, x) in enumerate(b.items()):
        exp = 1.0 if (gate is None or gate.asof(d) == True) else 0.0
        if use_vt and exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k]); exp *= float(np.clip(tgt / rv, 0, CAP)) if rv > 1e-9 else CAP
        out.append(exp * x)
    return pd.Series(out, index=b.index)


def stat(b, hold):
    r = b.values
    if len(r) == 0 or not np.all(np.isfinite(r)):
        return {"annual": 0, "sharpe": 0, "mdd": 0}
    rpy = 252 / hold; eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (rpy / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(rpy)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd}


def main() -> int:
    panel = pd.read_parquet(CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    vol20 = close.pct_change().rolling(20, min_periods=10).std()
    ks = fdr.DataReader("KS11", "2014-01-01", "2026-05-01")["Close"]; ks.index = pd.to_datetime(ks.index).normalize()
    gate = ks > ks.rolling(200, min_periods=100).mean()
    logger.info(f"KOSPI: {close.shape[1]} ticker, {close.shape[0]} 거래일, KS11 ok")

    logger.info("=" * 76)
    logger.info("KOSPI 엣지 탐색 (full-cycle, raw=gate없음)")
    logger.info("=" * 76)
    results = {}
    grid = [("reversion", 5, 5), ("reversion", 10, 5), ("reversion", 10, 10),
            ("reversion", 20, 10), ("reversion", 20, 20),
            ("lowvol", 0, 20), ("momentum", 60, 20)]
    for kind, lb, hold in grid:
        b = basket(close, dvol, vol20, lb, hold, kind)
        s = stat(b, hold)
        results[f"{kind}_lb{lb}_h{hold}"] = s
        logger.info(f"  {kind:9s} lb{lb:<3d} h{hold:<3d}: annual={s['annual']:+.1%}  Sharpe={s['sharpe']:+.2f}  MDD={s['mdd']:.1%}")

    best_key = max(results, key=lambda k: results[k]["sharpe"])
    bk, blb, bh = best_key.split("_")
    blb = int(blb[2:]); bh = int(bh[1:])
    logger.info(f"\nbest raw: {best_key} Sharpe={results[best_key]['sharpe']:.2f}")
    bb = basket(close, dvol, vol20, blb, bh, bk)
    for label, g, vt in [("+gate", gate, False), ("+gate+voltarget", gate, True)]:
        s = stat(overlay(bb, g, bh, vt), bh)
        results[f"BEST{label}"] = s
        logger.info(f"  {best_key} {label}: annual={s['annual']:+.1%}  Sharpe={s['sharpe']:+.2f}  MDD={s['mdd']:.1%}")

    logger.info("\nVERDICT:")
    top = max(results.items(), key=lambda kv: kv[1]["sharpe"])
    if top[1]["sharpe"] >= 0.5:
        logger.info(f"  배포가능 후보: {top[0]} Sharpe={top[1]['sharpe']:.2f} → KOSPI 엔진 설계 진행")
    else:
        logger.info(f"  best={top[0]} Sharpe={top[1]['sharpe']:.2f} < 0.5 → KOSPI는 가격신호로 배포불가 (펀더멘털/패스 필요)")
    logger.info("=" * 76)

    out = Path("v3/research/reports/korea_kospi_edge.json")
    out.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                               "results": results}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
