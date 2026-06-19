"""제안 #2 — 하드 스파시티 게이트 OOS probe (NASDAQ, 2026-06-19).

제안: ic_to_weights 의 0.10 floor 제거 + 통계 미달(IC<0.02) 알파 가중치 0 강제
(winner-take-most). 가설: 죽은 알파의 floor 보장이 유효 알파 부호를 희석.

반박 prior(낮음): Edge 실패 근본=NASDAQ 방향엣지 ~0(floor 희석 아님), 하드컷은 월별
weight 불안정 재도입. in-sample IC 검증=순환 → **walk-forward OOS** 로 검증.

방법 (무조건부, 4 DEFAULT directional alpha):
  1. ohlcv → 5일 step 패널: alpha 값 + 5d forward excess return.
  2. walk-forward: train(50 표본일≈1y) IC → 두 가중치:
       floored  = ic_to_weights(IC)  [production: shrink+sqrt+0.10 floor]
       sparsity = IC<0.02 → 0, 생존 알파 shrunk 비례 (no floor, winner-take-most)
     → test(4 표본일≈1mo) 에 direction=Σw·α 적용 → OOS IC(direction, fwd_ret).
  3. floored vs sparsity OOS direction IC + 가중치 turnover(L1, 불안정성).

판정: sparsity OOS IC 가 floored 를 유의하게 상회해야 채택. 아니면 기각(prior 확인).

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/nasdaq_sparsity_probe.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

from v3.data.feature_engineer import VolFeatureEngineer
from v3.strategy.alpha_sources import DEFAULT_DIRECTIONAL, compute_directional

MIN_VANILLA_IC = 0.02
STEP, FWD = 5, 5                  # 5거래일 step, 5일 forward
TRAIN_N, TEST_N = 50, 4          # 표본일 단위 (~1y train, ~1mo test)
SLICE_DAYS = 400                 # alpha 계산용 trailing 윈도우
REPORTS = Path("v3/research/reports")
OHLCV = Path("v3/data/raw/ohlcv_raw.parquet")
ALPHAS = [s.name for s in DEFAULT_DIRECTIONAL]


def ic_to_weights_floored(ic: dict, min_weight=0.10) -> dict:
    """production ic_to_weights 복제 (shrink→sqrt→0.10 floor)."""
    if not ic:
        return {}
    n = len(ic)
    if n * min_weight >= 1.0:
        return {a: 1.0 / n for a in ic}
    shrunk = {a: max(v - MIN_VANILLA_IC, 0.0) for a, v in ic.items()}
    if sum(shrunk.values()) <= 1e-9:
        return {a: 1.0 / n for a in ic}
    smoothed = {a: math.sqrt(s) for a, s in shrunk.items()}
    total = sum(smoothed.values())
    free = 1.0 - n * min_weight
    return {a: min_weight + free * smoothed[a] / total for a in ic}


def weights_sparsity(ic: dict) -> dict:
    """하드 스파시티: IC<0.02 → 0, 생존 알파 shrunk 비례 (no floor, winner-take-most)."""
    shrunk = {a: max(v - MIN_VANILLA_IC, 0.0) for a, v in ic.items()}
    total = sum(shrunk.values())
    if total <= 1e-9:
        return {a: 0.0 for a in ic}          # 전부 미달 → 신호 없음(현금)
    return {a: shrunk[a] / total for a in ic}


def build_panel(feat: pd.DataFrame) -> pd.DataFrame:
    close_wide = feat.pivot_table(index="date", columns="ticker", values="close",
                                  aggfunc="last").sort_index()
    all_dates = list(close_wide.index)
    d2i = {d: i for i, d in enumerate(all_dates)}
    sample = all_dates[::STEP]
    rows = []
    for k, t in enumerate(sample):
        ti = d2i[t]
        if ti + FWD >= len(all_dates):
            break
        tf = all_dates[ti + FWD]
        sl = feat[(feat["date"] > t - pd.Timedelta(days=SLICE_DAYS)) & (feat["date"] <= t)]
        if sl["date"].nunique() < 60:
            continue
        d = compute_directional(sl)                       # ticker × alpha
        if d.empty:
            continue
        fwd = (close_wide.loc[tf] / close_wide.loc[t] - 1.0).dropna()
        if len(fwd) < 10:
            continue
        excess = fwd - fwd.mean()
        j = d.copy()
        j["excess"] = excess.reindex(j.index)
        j["date"] = t
        j = j.dropna(subset=["excess"] + ALPHAS)
        if len(j):
            rows.append(j.reset_index())
        if (k + 1) % 40 == 0:
            logger.info(f"  panel {k+1}/{len(sample)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def vanilla_ic(df: pd.DataFrame) -> dict:
    out = {}
    for a in ALPHAS:
        m = df[a].notna() & df["excess"].notna()
        if m.sum() < 50:
            out[a] = 0.0; continue
        rho, _ = spearmanr(df.loc[m, a], df.loc[m, "excess"])
        out[a] = float(rho) if rho is not None else 0.0
    return out


def direction_ic(df: pd.DataFrame, w: dict) -> float:
    if sum(abs(v) for v in w.values()) < 1e-9:
        return np.nan                        # 신호 없음 (sparsity 전부 0)
    sig = sum(w[a] * df[a] for a in ALPHAS)
    m = sig.notna() & df["excess"].notna()
    if m.sum() < 30:
        return np.nan
    rho, _ = spearmanr(sig[m], df.loc[m, "excess"])
    return float(rho) if rho is not None else np.nan


def main() -> int:
    ohlcv = pd.read_parquet(OHLCV)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    logger.info(f"OHLCV {ohlcv['ticker'].nunique()} tickers {ohlcv['date'].min().date()}~{ohlcv['date'].max().date()}")
    feat = VolFeatureEngineer().compute_all(ohlcv)
    logger.info("패널 구축...")
    panel = build_panel(feat)
    if panel.empty:
        logger.error("empty panel"); return 1
    dates = sorted(panel["date"].unique())
    logger.info(f"패널: {len(panel)} rows, {len(dates)} 표본일 {pd.Timestamp(dates[0]).date()}~{pd.Timestamp(dates[-1]).date()}")

    full_ic = vanilla_ic(panel)
    logger.info(f"전체기간 vanilla IC: " + "  ".join(f"{a}={v:+.3f}" for a, v in full_ic.items()))

    # walk-forward
    fold_floored, fold_sparsity = [], []
    w_floored_prev, w_sparsity_prev = None, None
    turn_floored, turn_sparsity = [], []
    cash_folds = 0
    i = TRAIN_N
    while i + TEST_N <= len(dates):
        tr_dates = set(dates[i - TRAIN_N:i]); te_dates = set(dates[i:i + TEST_N])
        tr = panel[panel["date"].isin(tr_dates)]; te = panel[panel["date"].isin(te_dates)]
        ic = vanilla_ic(tr)
        wf = ic_to_weights_floored(ic); ws = weights_sparsity(ic)
        icf, ics = direction_ic(te, wf), direction_ic(te, ws)
        if not np.isnan(icf):
            fold_floored.append(icf)
        if np.isnan(ics) or sum(abs(v) for v in ws.values()) < 1e-9:
            cash_folds += 1
        else:
            fold_sparsity.append(ics)
        if w_floored_prev:
            turn_floored.append(sum(abs(wf[a] - w_floored_prev[a]) for a in ALPHAS))
            turn_sparsity.append(sum(abs(ws[a] - w_sparsity_prev[a]) for a in ALPHAS))
        w_floored_prev, w_sparsity_prev = wf, ws
        i += TEST_N

    n_folds = len(fold_floored)
    mf, ms = np.mean(fold_floored), (np.mean(fold_sparsity) if fold_sparsity else np.nan)
    logger.info("=" * 78)
    logger.info(f"walk-forward OOS direction IC ({n_folds} folds, train{TRAIN_N}/test{TEST_N} 표본일):")
    logger.info(f"  floored (production)  : mean {mf:+.4f}  std {np.std(fold_floored):.4f}")
    logger.info(f"  sparsity (제안 #2)     : mean {ms:+.4f}  std {np.std(fold_sparsity):.4f}  "
                f"(현금 fold {cash_folds} = 전 알파 미달)")
    logger.info(f"  가중치 turnover (L1, 불안정성): floored {np.mean(turn_floored):.3f}  "
                f"sparsity {np.mean(turn_sparsity):.3f}")
    logger.info("")
    diff = (ms - mf) if not np.isnan(ms) else -1
    logger.info("VERDICT:")
    logger.info(f"  · OOS IC: sparsity {ms:+.4f} vs floored {mf:+.4f} → "
                f"{'sparsity 우위' if diff > 0.005 else ('무차이' if abs(diff) <= 0.005 else 'sparsity 열위')}")
    logger.info(f"  · 안정성: sparsity turnover {np.mean(turn_sparsity):.3f} vs floored "
                f"{np.mean(turn_floored):.3f} → {'sparsity 더 불안정' if np.mean(turn_sparsity)>np.mean(turn_floored) else '비슷/안정'}")
    accept = (diff > 0.005)
    logger.info(f"  → 채택? {'YES (희석 가설 입증)' if accept else 'NO (floor 유지 — prior 확인)'}")
    logger.info("=" * 78)

    p = REPORTS / "nasdaq_sparsity_probe.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "full_vanilla_ic": full_ic, "n_folds": n_folds,
        "oos_ic": {"floored_mean": float(mf), "sparsity_mean": float(ms) if not np.isnan(ms) else None,
                   "cash_folds": cash_folds},
        "turnover": {"floored": float(np.mean(turn_floored)), "sparsity": float(np.mean(turn_sparsity))},
        "accept": bool(accept),
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
