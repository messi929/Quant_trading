"""한국 엔진 return-side sweep — leverage cap + 신호변형 (V4, 2026-05-30).

full-cycle 최선이 KOSDAQ momentum+gate+voltarget(cap1.0) = +7.3%/0.50/MDD-21%.
vol-target cap1.0은 '내리기만' 해서 return 깎임. 제대로 된 constant-vol scaling
(Barroso)은 평시 레버리지로 target vol 맞춤 → return 회복. + 신호 변형 탐색.

  Sweep A (cap): lb60/hold20/N20 고정, cap {1.0,1.25,1.5,2.0,3.0}. 이론근거 lever.
  Sweep B (signal): cap1.5 고정, lookback {40,60,90,120} x N {15,20,30}. smoothness로
    과적합 경계 (best 선택 신뢰 말 것).

긴 캐시 재사용. gate=sma200, trailing 없음(기각됨). 상폐/비용 동일.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_lever_sweep.py
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

LIQ_TOP = 100
COST, DELIST_PEN = 0.004, 0.5
DATA_START, END = "2014-01-01", "2026-05-01"
TARGET_VOL, VOL_WIN = 0.15, 6
REPORTS = Path("v3/research/reports")
MARKETS = {"KOSDAQ": ("korea_kosdaq_long_cache.parquet", "KQ11"),
           "KOSPI": ("korea_kospi_long_cache.parquet", "KS11")}


def pos_ret(path, entry):
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    if lv is not None and path[lv] > 0:
        return path[lv] / entry - 1.0 - DELIST_PEN
    return -1.0


def basket(close, dvol, lookback, hold, n_pos) -> pd.Series:
    dates = close.index.tolist(); out = {}
    for i in range(lookback, len(dates) - hold, hold):
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[dates[i]] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - lookback][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            out[dates[i]] = 0.0; continue
        picks = trend.nlargest(min(n_pos, len(trend))).index
        pr = [pos_ret(close.iloc[i:i + hold + 1][t], close.iloc[i][t]) for t in picks]
        out[dates[i]] = float(np.mean(pr)) - COST
    return pd.Series(out).sort_index()


def overlay(b: pd.Series, gate: pd.Series, cap: float, rpy: float) -> pd.Series:
    tgt = TARGET_VOL / np.sqrt(rpy); vals = b.values; rets = []
    for k, (d, x) in enumerate(b.items()):
        g = gate.asof(d); exp = 1.0 if (g is True or g == True) else 0.0
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, cap)) if rv > 1e-9 else cap
        rets.append(exp * x)
    return pd.Series(rets, index=b.index)


def stat(r, rpy):
    r = np.asarray(r)
    if len(r) == 0:
        return {"annual": 0, "sharpe": 0, "mdd": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (rpy / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(rpy)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd}


def sma200(index):
    return index > index.rolling(200, min_periods=100).mean()


def main() -> int:
    out_all = {}
    for m, (cfile, icode) in MARKETS.items():
        panel = pd.read_parquet(REPORTS / cfile)
        close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
        dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
        index = fdr.DataReader(icode, DATA_START, END)["Close"]; index.index = pd.to_datetime(index.index).normalize()
        gate = sma200(index)
        out_all[m] = {"cap_sweep": {}, "signal_sweep": {}}

        logger.info("=" * 82)
        logger.info(f"{m} — Sweep A: leverage cap (lb60/hold20/N20, gate=sma200)")
        logger.info("=" * 82)
        rpy = 252 / 20
        b_base = basket(close, dvol, 60, 20, 20)
        for cap in [1.0, 1.25, 1.5, 2.0, 3.0]:
            st = stat(overlay(b_base, gate, cap, rpy).values, rpy)
            out_all[m]["cap_sweep"][cap] = st
            logger.info(f"  cap={cap:<4}: annual={st['annual']:+.1%}  Sharpe={st['sharpe']:+.2f}  MDD={st['mdd']:.1%}")

        logger.info(f"{m} — Sweep B: signal (cap1.5, hold20)  lookback x N")
        for lb in [40, 60, 90, 120]:
            row = []
            for n in [15, 20, 30]:
                bb = basket(close, dvol, lb, 20, n)
                st = stat(overlay(bb, gate, 1.5, rpy).values, rpy)
                out_all[m]["signal_sweep"][f"lb{lb}_N{n}"] = st
                row.append(f"N{n}: Sh={st['sharpe']:+.2f}/an={st['annual']:+.0%}/MDD{st['mdd']:.0%}")
            logger.info(f"  lb{lb:<4}: " + "   ".join(row))
        logger.info("")

    logger.info("VERDICT:")
    for m in MARKETS:
        allc = {**{f"cap{k}": v for k, v in out_all[m]["cap_sweep"].items()}, **out_all[m]["signal_sweep"]}
        best = max(allc.items(), key=lambda kv: kv[1]["sharpe"])
        logger.info(f"  {m}: best={best[0]} Sharpe={best[1]['sharpe']:.2f} annual={best[1]['annual']:+.1%} MDD={best[1]['mdd']:.1%}")
    logger.info("=" * 82)

    p = REPORTS / "korea_lever_sweep.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "markets": out_all}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
