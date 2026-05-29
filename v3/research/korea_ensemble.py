"""한국 multi-lookback momentum ensemble — fragility 제거 (V4, 2026-05-30).

return-side sweep 에서 단일 lookback 이 fragile 확인 (lb90: 2021-26 best 1.18 →
full-cycle worst 음수, lb40 0.92 도 신뢰 불가). 해법 = 단일 lb 베팅 대신 multi-lb
ensemble — lookback {40,60,90,120} cross-sectional 랭크 평균으로 종목 선정.
한 horizon 의 운/불운에 안 휘둘림 → robust.

신호: 각 rebalance 시점 PIT 거래대금 top100 pool 에서
  - 각 lb 의 past return cross-sectional 랭크 → 4개 랭크 평균 = blended momentum
  - TS trend filter: 4개 lb past return 평균 > 0 (추세)
  - blended 랭크 상위 N long
gate(sma200) + vol-target(cap1.5) overlay. 긴 캐시 재사용.

비교: 개별 lb sleeve vs ensemble. ensemble 이 개별 평균보다 안정적이고 0.5~0.6 robust 인지.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_ensemble.py
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

LBS = [40, 60, 90, 120]
MAXLB, HOLD, LIQ_TOP, N_POS = 120, 20, 100, 20
COST, DELIST_PEN = 0.004, 0.5
DATA_START, END = "2014-01-01", "2026-05-01"
TARGET_VOL, VOL_WIN, CAP = 0.15, 6, 1.5
RPY = 252 / HOLD
REPORTS = Path("v3/research/reports")
MARKETS = {"KOSDAQ": ("korea_kosdaq_long_cache.parquet", "KQ11"),
           "KOSPI": ("korea_kospi_long_cache.parquet", "KS11")}
CRASHES = {"2018 Q4": ("2018-06-01", "2019-01-31"),
           "2020 COVID": ("2020-01-01", "2020-05-31"),
           "2022 약세장": ("2022-01-01", "2022-12-31")}


def pos_ret(path, entry):
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    if lv is not None and path[lv] > 0:
        return path[lv] / entry - 1.0 - DELIST_PEN
    return -1.0


def basket_single(close, dvol, lb) -> pd.Series:
    dates = close.index.tolist(); out = {}
    for i in range(MAXLB, len(dates) - HOLD, HOLD):
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[dates[i]] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            out[dates[i]] = 0.0; continue
        picks = trend.nlargest(min(N_POS, len(trend))).index
        out[dates[i]] = float(np.mean([pos_ret(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t]) for t in picks])) - COST
    return pd.Series(out).sort_index()


def basket_ensemble(close, dvol) -> pd.Series:
    dates = close.index.tolist(); out = {}
    for i in range(MAXLB, len(dates) - HOLD, HOLD):
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[dates[i]] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        pasts = {lb: (close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0) for lb in LBS}
        df = pd.DataFrame(pasts).dropna()
        if len(df) < 20:
            out[dates[i]] = 0.0; continue
        mean_past = df.mean(axis=1)                 # TS trend filter
        blended = df.rank().mean(axis=1)            # cross-sectional 랭크 평균
        cand = blended[mean_past > 0]
        if len(cand) == 0:
            out[dates[i]] = 0.0; continue
        picks = cand.nlargest(min(N_POS, len(cand))).index
        out[dates[i]] = float(np.mean([pos_ret(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t]) for t in picks])) - COST
    return pd.Series(out).sort_index()


def overlay(b, gate, cap=CAP):
    tgt = TARGET_VOL / np.sqrt(RPY); vals = b.values; rets = []
    for k, (d, x) in enumerate(b.items()):
        g = gate.asof(d); exp = 1.0 if (g is True or g == True) else 0.0
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, cap)) if rv > 1e-9 else cap
        rets.append(exp * x)
    return pd.Series(rets, index=b.index)


def stat(r):
    r = np.asarray(r)
    if len(r) == 0:
        return {"annual": 0, "sharpe": 0, "mdd": 0, "ret": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "ret": tot}


def sub(s, ps, pe):
    return stat(s[(s.index >= ps) & (s.index <= pe)].values)


def main() -> int:
    out_all = {}
    for m, (cfile, icode) in MARKETS.items():
        panel = pd.read_parquet(REPORTS / cfile)
        close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
        dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
        index = fdr.DataReader(icode, DATA_START, END)["Close"]; index.index = pd.to_datetime(index.index).normalize()
        gate = index > index.rolling(200, min_periods=100).mean()

        logger.info("=" * 80)
        logger.info(f"{m} — multi-lb ensemble (gate+vol-target cap{CAP}, full-cycle)")
        logger.info("=" * 80)
        # 개별 lb (참고)
        singles = {}
        logger.info("개별 lb sleeve:")
        for lb in LBS:
            s = overlay(basket_single(close, dvol, lb), gate)
            st = stat(s.values); singles[lb] = st
            logger.info(f"  lb{lb:<4d}: Sharpe={st['sharpe']:+.2f}  annual={st['annual']:+.1%}  MDD={st['mdd']:.1%}")
        mean_single_sh = float(np.mean([v["sharpe"] for v in singles.values()]))

        # ensemble
        ens = overlay(basket_ensemble(close, dvol), gate)
        est = stat(ens.values)
        crashes = {c: sub(ens, ps, pe) for c, (ps, pe) in CRASHES.items()}
        logger.info(f"ENSEMBLE: Sharpe={est['sharpe']:+.2f}  annual={est['annual']:+.1%}  MDD={est['mdd']:.1%}  "
                    f"(개별 평균 Sharpe={mean_single_sh:+.2f})")
        logger.info("  crashes [MDD/ret]: " + "  ".join(f"{c}={v['mdd']:+.0%}/{v['ret']:+.0%}" for c, v in crashes.items()))
        out_all[m] = {"singles": singles, "ensemble": est, "ensemble_crashes": crashes,
                      "mean_single_sharpe": mean_single_sh}
        logger.info("")

    logger.info("VERDICT (ensemble robust 개선?):")
    for m in MARKETS:
        e = out_all[m]["ensemble"]; ms = out_all[m]["mean_single_sharpe"]
        verdict = "개선" if e["sharpe"] > ms + 0.05 else ("안정화" if e["sharpe"] >= ms - 0.05 else "약화")
        logger.info(f"  {m}: ensemble Sharpe={e['sharpe']:.2f} vs 개별평균 {ms:.2f} → {verdict}  "
                    f"(annual {e['annual']:+.1%}, MDD {e['mdd']:.1%})")
    logger.info("=" * 80)

    p = REPORTS / "korea_ensemble.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "lbs": LBS, "cap": CAP, "markets": out_all}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
