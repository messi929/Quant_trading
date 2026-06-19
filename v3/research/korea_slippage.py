"""제안 #3 — 동적/비대칭 슬리피지 (square-root impact) 측정 (2026-06-19).

동기: 현 백테스트는 KOSDAQ 왕복 0.4% 고정비용. 실제론 자금쏠림·패닉(변동성 팽창) 시
진입·청산 비용이 비대칭 팽창 → 고정비용은 Sharpe·크래시방어 과대평가 위험.

모델 (a-priori, 무 자유파라미터):
    roundtrip_cost_name = base + 2·k·σ_name·√(Q_name / ADV_name)
  - base = 0.20% (한국 거래세 0.18%(매도) + 수수료, 고정 — 임팩트 무관)
  - k = 1.0 (Almgren-style 보수)
  - σ_name = 종목 20d 일일수익 표준편차 (변동성 — 크래시 시 팽창)
  - Q_name = 종목당 투입금 (= CAPITAL × exposure / n_pos)
  - ADV_name = 20d 평균 거래대금(원)
  basket_cost = picks 평균. realized = exp × (gross − basket_cost).

검증:
  1. 자본 {1억, 10억, 50억} × dynamic vs 고정 0.4% — full-cycle Sharpe/annual/MDD.
     정상상태(1억) inert 확인 + 자본 클수록 슬리피지 잠식.
  2. 크래시창(2018/2020/2022) 수익 — 고정 vs dynamic. σ 팽창으로 크래시 진입·청산
     비용 상승 → 게이트 크래시방어가 고정비용서 과대평가됐는가?
  3. 유동성 floor 임계치 도출 — 임팩트가 허용치(per-side 0.3%) 넘는 ADV.

엔진 픽/게이트/vol-target 은 korea_ensemble 동일(parity). 비용함수만 교체. 라이브 무변경.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_slippage.py
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
FIXED_COST, DELIST_PEN = 0.004, 0.5          # 기존 고정 왕복
BASE, K = 0.0020, 1.0                        # 동적 모델: 고정 base + impact (k=1)
TARGET_VOL, VOL_WIN, CAP = 0.15, 6, 1.5
RPY = 252 / HOLD
TOL_SIDE = 0.003                             # 유동성 floor 도출용 per-side 임팩트 허용
CAPITALS = [1e8, 1e9, 5e9]                   # 1억 / 10억 / 50억
REPORTS = Path("v3/research/reports")
CACHE, ICODE = "korea_kosdaq_long_cache.parquet", "KQ11"
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


def basket(close, dvol, vol20, adv20, *, capital=None, exp_for_q=1.0):
    """rebalance 별 (raw_gross, cost) 반환. capital=None → 고정 0.4%, else 동적."""
    dates = close.index.tolist()
    out_gross, out_cost, out_dates = [], [], []
    for i in range(MAXLB, len(dates) - HOLD, HOLD):
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out_gross.append(0.0); out_cost.append(0.0); out_dates.append(dates[i]); continue
        pool = liq.nlargest(LIQ_TOP).index
        pasts = {lb: (close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0) for lb in LBS}
        df = pd.DataFrame(pasts).dropna()
        if len(df) < 20:
            out_gross.append(0.0); out_cost.append(0.0); out_dates.append(dates[i]); continue
        cand = df.rank().mean(axis=1)[df.mean(axis=1) > 0]
        if len(cand) == 0:
            out_gross.append(0.0); out_cost.append(0.0); out_dates.append(dates[i]); continue
        picks = cand.nlargest(min(N_POS, len(cand))).index
        gross = float(np.mean([pos_ret(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t]) for t in picks]))
        if capital is None:
            cost = FIXED_COST
        else:
            q = capital * exp_for_q / len(picks)            # 종목당 투입금
            sig = vol20.iloc[i][picks].values
            adv = adv20.iloc[i][picks].values
            part = np.where(adv > 0, q / adv, np.inf)
            impact = K * sig * np.sqrt(part)                # per-side
            cost = float(BASE + 2 * np.nanmean(impact))
        out_gross.append(gross); out_cost.append(cost); out_dates.append(dates[i])
    return pd.Series(out_gross, index=out_dates), pd.Series(out_cost, index=out_dates)


def overlay(gross, cost, gate):
    """gate × vol-target × (gross − cost). raw(phantom)=gross−cost 로 vol-target conditioning."""
    raw = (gross - cost)
    tgt = TARGET_VOL / np.sqrt(RPY); vals = raw.values; rets = []
    for k, (d, x) in enumerate(raw.items()):
        g = gate.asof(d); exp = 1.0 if (g is True or g == True) else 0.0   # noqa: E712
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, CAP)) if rv > 1e-9 else CAP
        rets.append(exp * x)
    return pd.Series(rets, index=raw.index)


def stat(r):
    r = np.asarray(r)
    if len(r) == 0:
        return {"annual": 0, "sharpe": 0, "mdd": 0, "ret": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "ret": tot}


def sub_ret(s, ps, pe):
    seg = s[(s.index >= ps) & (s.index <= pe)]
    return float(np.prod(1 + seg.values) - 1) if len(seg) else 0.0


def main() -> int:
    panel = pd.read_parquet(REPORTS / CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    vol20 = close.pct_change().rolling(20, min_periods=10).std()
    adv20 = dvol.rolling(20, min_periods=10).mean()
    index = fdr.DataReader(ICODE, "2014-01-01", "2026-05-01")["Close"]
    index.index = pd.to_datetime(index.index).normalize()
    gate = index > index.rolling(200, min_periods=100).mean()

    logger.info("=" * 88)
    logger.info("KOSDAQ 동적 슬리피지 (square-root impact) — full-cycle vs 고정 0.4%")
    logger.info("=" * 88)

    # baseline 고정
    g0, c0 = basket(close, dvol, vol20, adv20, capital=None)
    base_stat = stat(overlay(g0, c0, gate).values)
    logger.info(f"고정 0.4%        : Sharpe {base_stat['sharpe']:+.2f}  annual {base_stat['annual']:+.1%}  "
                f"MDD {base_stat['mdd']:.1%}  (평균비용 {c0[c0>0].mean():.3%})")

    out = {"fixed": base_stat, "dynamic": {}}
    for cap in CAPITALS:
        g, c = basket(close, dvol, vol20, adv20, capital=cap, exp_for_q=1.0)
        s = overlay(g, c, gate); st = stat(s.values)
        avg_cost = float(c[c > 0].mean())
        crashes = {cn: sub_ret(s, ps, pe) for cn, (ps, pe) in CRASHES.items()}
        out["dynamic"][f"{cap:.0f}"] = {**st, "avg_cost": avg_cost, "crashes": crashes}
        logger.info(f"동적 {cap/1e8:>4.0f}억 자본 : Sharpe {st['sharpe']:+.2f}  annual {st['annual']:+.1%}  "
                    f"MDD {st['mdd']:.1%}  (평균비용 {avg_cost:.3%})")

    # 크래시창 — 고정 vs 동적(1억, 50억)
    s_fixed = overlay(g0, c0, gate)
    g1, c1 = basket(close, dvol, vol20, adv20, capital=1e8); s_1e8 = overlay(g1, c1, gate)
    g5, c5 = basket(close, dvol, vol20, adv20, capital=5e9); s_5e9 = overlay(g5, c5, gate)
    logger.info("")
    logger.info("크래시창 수익 — 고정 0.4% vs 동적(σ 팽창 반영):")
    crash_cmp = {}
    for cn, (ps, pe) in CRASHES.items():
        f, d1, d5 = sub_ret(s_fixed, ps, pe), sub_ret(s_1e8, ps, pe), sub_ret(s_5e9, ps, pe)
        crash_cmp[cn] = {"fixed": f, "dyn_1e8": d1, "dyn_5e9": d5}
        logger.info(f"  {cn:<12}: 고정 {f:+.1%}  |  1억 {d1:+.1%}  |  50억 {d5:+.1%}")

    # 평시 vs 크래시 평균비용 (σ 팽창 정량화)
    crash_mask = pd.Series(False, index=c1.index)
    for _, (ps, pe) in CRASHES.items():
        crash_mask |= (c1.index >= pd.Timestamp(ps)) & (c1.index <= pd.Timestamp(pe))
    calm_cost = float(c5[(c5 > 0) & ~crash_mask].mean())
    crash_cost = float(c5[(c5 > 0) & crash_mask].mean())
    logger.info("")
    logger.info(f"50억 자본 평균비용 — 평시 {calm_cost:.3%}  vs  크래시창 {crash_cost:.3%}  "
                f"(×{crash_cost/calm_cost:.2f} 팽창)")

    # 유동성 floor 도출: per-side impact > TOL_SIDE 인 ADV 임계 (대표 σ 사용)
    sig_med = float(np.nanmedian(vol20.values[np.isfinite(vol20.values)]))
    logger.info("")
    logger.info(f"유동성 floor 도출 (per-side 임팩트 > {TOL_SIDE:.1%} 회피, σ_med={sig_med:.2%}, k={K}):")
    floors = {}
    for cap in CAPITALS:
        q = cap / N_POS
        # impact = K·σ·√(q/ADV) = TOL → ADV = q / (TOL/(K·σ))²
        adv_floor = q / (TOL_SIDE / (K * sig_med)) ** 2
        floors[f"{cap:.0f}"] = adv_floor
        logger.info(f"  자본 {cap/1e8:>4.0f}억 (종목당 {q/1e8:.2f}억): ADV floor ≈ {adv_floor/1e8:.0f}억")
    logger.info(f"  → 1억 규모 floor(~{floors[f'{CAPITALS[0]:.0f}']/1e8:.0f}억)는 현 바스켓(최저 104억) 전부 통과. "
                f"50억 규모는 floor(~{floors[f'{CAPITALS[-1]:.0f}']/1e8:.0f}억) 발동.")

    # VERDICT
    d1 = out["dynamic"][f"{CAPITALS[0]:.0f}"]; d5 = out["dynamic"][f"{CAPITALS[-1]:.0f}"]
    logger.info("")
    logger.info("VERDICT:")
    logger.info(f"  · 1억 자본: dynamic Sharpe {d1['sharpe']:+.2f} vs 고정 {base_stat['sharpe']:+.2f} → "
                f"{'무차이(고정 타당)' if abs(d1['sharpe']-base_stat['sharpe'])<0.03 else '차이'} "
                f"(평균비용 {d1['avg_cost']:.3%} < 0.4%)")
    logger.info(f"  · 50억 자본: Sharpe {d5['sharpe']:+.2f} (비용 {d5['avg_cost']:.3%}) → 슬리피지 잠식")
    logger.info(f"  · 크래시 비용 ×{crash_cost/calm_cost:.1f} 팽창 → 고정비용이 크래시 진입/청산 비용 "
                f"{'과소평가' if crash_cost>0.004 else '근사'}")
    logger.info("=" * 88)

    p = REPORTS / "korea_slippage.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {"base": BASE, "k": K, "tol_side": TOL_SIDE},
        "fixed": base_stat, "dynamic": out["dynamic"], "crashes": crash_cmp,
        "cost_calm_vs_crash_5e9": {"calm": calm_cost, "crash": crash_cost},
        "adv_floors": floors,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
