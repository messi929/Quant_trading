"""제안 #1 검증 — 트렌치 분할(window luck) 측정 (2026-06-19).

동기: V4 백테스트(korea_ensemble)는 rebalance 시작 offset=0 인 *단일 phase* 1개만
측정 → 보고된 Sharpe 0.66 에 '어느 날 평가했는가'(window luck)가 숨어있을 수 있음.
사용자 #1: 자본을 N 트렌치로 쪼개 시차 순환 → 진입·청산 시점 평활화로 타이밍 리스크
거세 (알파/게이트 무수정).

검증:
  1. HOLD(20) 개 phase offset 전부 단일운용 → Sharpe/annual/MDD 분산 = window luck 크기.
  2. N={4,5} 트렌치 일일 MTM 블렌드(균등 시차) → 블렌드 Sharpe/MDD vs phase 분포.
     블렌드가 (a) Sharpe ≈ phase 평균, (b) MDD·변동성 축소면 → window luck 거세 입증.

비용·게이트·vol-target·픽 수식은 korea_ensemble 과 동일(parity). 일일 MTM 로 확장만.
라이브 무변경 — 측정. (채택 시 운용은 자본 N등분·주 1회 1트렌치 리밸런스.)

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_tranche.py
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
TARGET_VOL, VOL_WIN, CAP = 0.15, 6, 1.5
ANN = 252.0
REPORTS = Path("v3/research/reports")
CACHE, ICODE = "korea_kosdaq_long_cache.parquet", "KQ11"


def picks_at(close, dvol, i):
    """i 시점 ensemble 픽 (korea_ensemble 와 동일). 없으면 ()."""
    liq = dvol.iloc[i].dropna()
    if len(liq) < 20:
        return ()
    pool = liq.nlargest(LIQ_TOP).index
    pasts = {lb: (close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0) for lb in LBS}
    df = pd.DataFrame(pasts).dropna()
    if len(df) < 20:
        return ()
    cand = df.rank().mean(axis=1)[df.mean(axis=1) > 0]
    if len(cand) == 0:
        return ()
    return tuple(cand.nlargest(min(N_POS, len(cand))).index)


def pos_ret_period(close, i, picks):
    """리밸런스 i 의 바스켓 20d 기간수익(상폐 패널티 포함) — vol-target conditioning 용."""
    rs = []
    for t in picks:
        path = close.iloc[i:i + HOLD + 1][t]; entry = close.iloc[i][t]
        final = path.iloc[-1]
        if pd.notna(final):
            rs.append(final / entry - 1.0)
        else:
            lv = path.last_valid_index()
            rs.append(path[lv] / entry - 1.0 - DELIST_PEN if (lv is not None and path[lv] > 0) else -1.0)
    return float(np.mean(rs)) - COST if rs else 0.0


def phase_daily(close, dvol, gate, offset) -> pd.Series:
    """단일 phase(시작 offset) 의 일일 수익 Series (gate×vol-target×equal-weight 픽, 일일 MTM)."""
    dates = close.index
    ridx = [i for i in range(MAXLB + offset, len(dates) - HOLD, HOLD)]
    period_rets, picks_list = [], []
    for i in ridx:
        pk = picks_at(close, dvol, i)
        picks_list.append((i, pk))
        period_rets.append(pos_ret_period(close, i, pk) if pk else 0.0)

    tgt = TARGET_VOL / np.sqrt(ANN / HOLD)
    daily = pd.Series(0.0, index=dates)
    for k, (i, pk) in enumerate(picks_list):
        g = gate.asof(dates[i]); exp = 1.0 if (g is True or g == True) else 0.0  # noqa: E712
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(period_rets[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, CAP)) if rv > 1e-9 else CAP
        if exp <= 0 or not pk:
            continue
        win = close.iloc[i:i + HOLD + 1][list(pk)]
        dret = win.pct_change().iloc[1:].mean(axis=1)  # 픽 일일 평균 수익률
        seg = exp * dret.fillna(0.0)
        seg.iloc[0] = seg.iloc[0] - COST              # 진입일 왕복비용
        daily.loc[seg.index] = seg.values
    return daily


def stat_daily(r: pd.Series):
    r = r[r.index >= r.index[(r != 0).argmax()]]  # 첫 비현금일부터
    rv = r.values
    if len(rv) == 0 or rv.std() < 1e-12:
        return {"annual": 0.0, "sharpe": 0.0, "mdd": 0.0}
    eq = np.cumprod(1 + rv); tot = float(eq[-1] - 1)
    yrs = len(rv) / ANN
    ann = float((1 + tot) ** (1 / yrs) - 1) if (tot > -1 and yrs > 0) else -1.0
    sh = float(rv.mean() / rv.std() * np.sqrt(ANN))
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd}


def main() -> int:
    panel = pd.read_parquet(REPORTS / CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    index = fdr.DataReader(ICODE, "2014-01-01", "2026-05-01")["Close"]
    index.index = pd.to_datetime(index.index).normalize()
    gate = index > index.rolling(200, min_periods=100).mean()

    logger.info("=" * 84)
    logger.info("제안 #1 — 트렌치 분할(window luck) 검증 — KOSDAQ full-cycle, 일일 MTM")
    logger.info("=" * 84)

    # 1. 전 phase offset
    phases = {p: phase_daily(close, dvol, gate, p) for p in range(HOLD)}
    pstats = {p: stat_daily(s) for p, s in phases.items()}
    shs = np.array([v["sharpe"] for v in pstats.values()])
    anns = np.array([v["annual"] for v in pstats.values()])
    mdds = np.array([v["mdd"] for v in pstats.values()])
    logger.info(f"phase offset {HOLD}개 단일운용 분산 (window luck):")
    logger.info(f"  Sharpe : min {shs.min():+.2f}  mean {shs.mean():+.2f}  max {shs.max():+.2f}  "
                f"std {shs.std():.2f}  (스프레드 {shs.max()-shs.min():.2f})")
    logger.info(f"  annual : min {anns.min():+.1%}  mean {anns.mean():+.1%}  max {anns.max():+.1%}")
    logger.info(f"  MDD    : best {mdds.max():.1%}  mean {mdds.mean():.1%}  worst {mdds.min():.1%}")
    logger.info(f"  (offset=0 = 현 백테스트 보고치: Sharpe {pstats[0]['sharpe']:+.2f} / "
                f"annual {pstats[0]['annual']:+.1%} / MDD {pstats[0]['mdd']:.1%})")
    logger.info("")

    # 2. 트렌치 블렌드
    logger.info("트렌치 블렌드 (균등 시차, 일일 1/N MTM 평균):")
    tr_out = {}
    for N in (4, 5):
        offs = [int(round(k * HOLD / N)) % HOLD for k in range(N)]
        blend = sum(phases[o] for o in offs) / N
        st = stat_daily(blend); tr_out[N] = {"offsets": offs, **st}
        logger.info(f"  N={N} (offsets {offs}): Sharpe {st['sharpe']:+.2f}  annual {st['annual']:+.1%}  "
                    f"MDD {st['mdd']:.1%}")
    logger.info("")

    # VERDICT
    best = tr_out[5] if tr_out[5]["sharpe"] >= tr_out[4]["sharpe"] else tr_out[4]
    bN = 5 if best is tr_out[5] else 4
    smooth = best["sharpe"] >= shs.mean() - 0.02            # 평균 수준 유지
    mdd_better = best["mdd"] > mdds.mean() + 0.005          # MDD 평균보다 개선(덜 깊음)
    luck_big = (shs.max() - shs.min()) >= 0.20
    logger.info("VERDICT:")
    logger.info(f"  · window luck 스프레드 Sharpe {shs.max()-shs.min():.2f} → "
                f"{'유의(트렌치 정당)' if luck_big else '작음(타이밍운 미미)'}")
    logger.info(f"  · 트렌치 N={bN}: Sharpe {best['sharpe']:+.2f} (phase 평균 {shs.mean():+.2f}), "
                f"MDD {best['mdd']:.1%} (phase 평균 {mdds.mean():.1%})")
    logger.info(f"  · 블렌드가 Sharpe 유지({smooth}) + MDD 개선({mdd_better})")
    accept = luck_big and smooth and mdd_better
    logger.info(f"  → 채택? {'YES — 트렌치가 window luck 거세 + 리스크 평활' if accept else 'NO / 부분'}")
    logger.info("=" * 84)

    p = REPORTS / "korea_tranche.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase_stats": {str(k): v for k, v in pstats.items()},
        "phase_sharpe": {"min": float(shs.min()), "mean": float(shs.mean()),
                         "max": float(shs.max()), "std": float(shs.std())},
        "tranche": tr_out, "accept": accept,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
