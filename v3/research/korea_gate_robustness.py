"""한국 regime gate robustness 검증 (V4, 2026-05-29).

korea_risk_overlay.py 에서 KOSDAQ+SMA200 = Sharpe1.02/MDD-24.7% 목표 통과했으나
12조합 중 best 선택 + n=62 소표본 → 과적합 의심. 3각 점검:

  (A) 파라미터 neighborhood 안정성 — SMA window {100..300}, mom lookback {40..200}
      sweep. 효과가 broad plateau(robust)인지 isolated spike(과적합)인지.
  (B) sub-period 일관성 — baseline vs gate 의 MDD/return 을 calendar sub-period별로.
      특히 2022 하락기에 gate 가 실제로 MDD 를 줄였는지 (regime filter 의 본질).
  (C) walk-forward — 과거 데이터로 best-Sharpe gate 선택 → 다음 해 적용 (expanding).
      WF 결과가 fixed-sma200 / no-gate 대비 어떤지. in-sample 선택 편향 제거.

PIT 캐시 재사용. 신호/비용/상폐처리 동일.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_gate_robustness.py
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

LOOKBACK, HOLD, LIQ_TOP, N_POS = 60, 20, 100, 20
COST, DELIST_PEN = 0.004, 0.5
START, END = "2021-01-01", "2026-05-01"
RPY = 252 / HOLD
REPORTS = Path("v3/research/reports")
MARKETS = {"KOSDAQ": ("korea_kosdaq_pit_cache.parquet", "KQ11"),
           "KOSPI": ("korea_kospi_pit_cache.parquet", "KS11")}


def gate_series(index: pd.Series, kind: str, param: int) -> pd.Series:
    if kind == "none":
        return pd.Series(True, index=index.index)
    if kind == "sma":
        return index > index.rolling(param, min_periods=param // 2).mean()
    if kind == "mom":
        return (index / index.shift(param) - 1.0) > 0
    raise ValueError(kind)


def sleeve(close, dvol, gate: pd.Series) -> pd.Series:
    dates = close.index.tolist()
    out = {}
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        d = dates[i]
        g = gate.asof(d)
        if not (g is True or g == True):
            out[d] = 0.0; continue
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[d] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - LOOKBACK][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            out[d] = 0.0; continue
        picks = trend.nlargest(min(N_POS, len(trend))).index
        pr = []
        for t in picks:
            e = close.iloc[i][t]; x = close.iloc[i + HOLD][t]
            if pd.notna(x):
                pr.append(x / e - 1.0)
            else:
                w = close.iloc[i:i + HOLD + 1][t]; lv = w.last_valid_index()
                pr.append((w[lv] / e - 1.0 - DELIST_PEN) if lv is not None and w[lv] > 0 else -1.0)
        out[d] = float(np.mean(pr)) - COST
    return pd.Series(out).sort_index()


def stat(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"annual": 0, "sharpe": 0, "mdd": 0, "n": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "n": len(r)}


def main() -> int:
    out_all = {}
    for mname, (cfile, icode) in MARKETS.items():
        panel = pd.read_parquet(REPORTS / cfile)
        close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
        dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
        index = fdr.DataReader(icode, START, END)["Close"]; index.index = pd.to_datetime(index.index).normalize()

        logger.info("=" * 78)
        logger.info(f"{mname} — gate robustness")
        logger.info("=" * 78)

        # (A) 파라미터 neighborhood 안정성
        logger.info("(A) 파라미터 안정성 (knife-edge면 과적합, plateau면 robust):")
        logger.info("  SMA window:")
        for w in [100, 150, 200, 250, 300]:
            s = stat(sleeve(close, dvol, gate_series(index, "sma", w)).values)
            logger.info(f"    sma{w:<4d}: Sharpe={s['sharpe']:+.2f}  MDD={s['mdd']:.1%}  annual={s['annual']:+.1%}")
        logger.info("  MOM lookback:")
        for lb in [40, 60, 90, 120, 200]:
            s = stat(sleeve(close, dvol, gate_series(index, "mom", lb)).values)
            logger.info(f"    mom{lb:<4d}: Sharpe={s['sharpe']:+.2f}  MDD={s['mdd']:.1%}  annual={s['annual']:+.1%}")

        # (B) sub-period 일관성 (baseline vs sma200)
        logger.info("(B) sub-period MDD/return — baseline(gate없음) vs sma200:")
        base = sleeve(close, dvol, gate_series(index, "none", 0))
        g200 = sleeve(close, dvol, gate_series(index, "sma", 200))
        subs = {"2021-22(하락포함)": ("2021-01-01", "2022-12-31"),
                "2023-24": ("2023-01-01", "2024-12-31"),
                "2025-26": ("2025-01-01", "2026-05-01")}
        for sp, (ps, pe) in subs.items():
            rb = base[(base.index >= ps) & (base.index <= pe)].values
            rg = g200[(g200.index >= ps) & (g200.index <= pe)].values
            sb, sg = stat(rb), stat(rg)
            logger.info(f"    {sp:16s}: base MDD={sb['mdd']:+.1%} ret={sb['annual']:+.1%}  |  "
                        f"sma200 MDD={sg['mdd']:+.1%} ret={sg['annual']:+.1%}")

        # (C) walk-forward — 과거로 best-Sharpe gate 선택 → 다음 해 적용
        cands = {"sma100": ("sma", 100), "sma150": ("sma", 150), "sma200": ("sma", 200),
                 "sma250": ("sma", 250), "mom40": ("mom", 40), "mom60": ("mom", 60),
                 "mom90": ("mom", 90), "mom120": ("mom", 120)}
        cand_ret = {k: sleeve(close, dvol, gate_series(index, t, p)) for k, (t, p) in cands.items()}
        all_dates = sorted(base.index)
        test_years = [2023, 2024, 2025]
        wf_rets, picks_log = [], []
        for ty in test_years:
            train_mask = [d for d in all_dates if d.year < ty]
            test_mask = [d for d in all_dates if d.year == ty or (ty == 2025 and d.year == 2026)]
            if len(train_mask) < 10 or not test_mask:
                continue
            # 과거(train) 구간 best Sharpe gate
            best_k, best_sh = "sma200", -9
            for k, sr in cand_ret.items():
                tr = sr[sr.index.isin(train_mask)].values
                sh = stat(tr)["sharpe"]
                if sh > best_sh:
                    best_sh, best_k = sh, k
            picks_log.append(f"{ty}:{best_k}")
            wf_rets.extend(cand_ret[best_k][cand_ret[best_k].index.isin(test_mask)].values.tolist())
        wf = stat(np.array(wf_rets))
        fixed = stat(g200[g200.index >= "2023-01-01"].values)
        nogate = stat(base[base.index >= "2023-01-01"].values)
        logger.info("(C) walk-forward (2023~, 과거로 gate 선택 → 다음해 적용):")
        logger.info(f"    WF 선택 이력: {picks_log}")
        logger.info(f"    WF       : Sharpe={wf['sharpe']:+.2f}  MDD={wf['mdd']:.1%}  annual={wf['annual']:+.1%}")
        logger.info(f"    fixed sma200 : Sharpe={fixed['sharpe']:+.2f}  MDD={fixed['mdd']:.1%}  annual={fixed['annual']:+.1%}")
        logger.info(f"    no-gate      : Sharpe={nogate['sharpe']:+.2f}  MDD={nogate['mdd']:.1%}  annual={nogate['annual']:+.1%}")
        logger.info("")
        out_all[mname] = {"wf": wf, "fixed_sma200": fixed, "no_gate": nogate, "wf_picks": picks_log}

    p = REPORTS / "korea_gate_robustness.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "markets": out_all}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
