"""P1 — 200d SMA 게이트 휩소 측정 + 대칭 히스테리시스(±1%) full-cycle 검증 (2026-06-19).

동기: 사용자가 코스닥 1,000 턱걸이에서 게이트 휩소(잦은 ON/OFF)로 자본이 비용에
녹는다며 ±1% 히스테리시스 버퍼 제안. 반박 가설:
  (a) 엔진은 20거래일 보유 → 게이트는 ~월 1회만 평가 (일간 톱니 아님).
  (b) 비용은 보유 기간당 부과·현금기간 무료 → 게이트 플립의 *한계* 비용 ≈ 0
      (바스켓이 어차피 매 리밸런스 전량 회전).
  (c) 상단 버퍼(SMA×1.01 진입)는 빠른 V자(COVID) 재진입을 지연 → 게이트의 알려진
      약점을 악화시킬 수 있음.
armchair 기각 대신 측정: 플립이 실제로 잦은가? 히스테리시스가 순효과로 이득인가?

측정:
  1. binary 게이트 플립 빈도 (full-cycle + danger band |idx/sma-1|<1% 내 플립).
  2. binary vs 대칭 히스테리시스(ON: ratio>1.01, OFF: ratio<0.99, 그 사이 prev 유지)
     full-cycle Sharpe/annual/MDD/전환수.
  3. 2020 COVID 등 crash window 참여(수익) 비교 — 상단버퍼 재진입 지연 검증.
  4. 전환 추가비용 민감도: 게이트 전환마다 extra cost 0/0.2/0.4% 부과 시 binary vs
     hysteresis 격차 (휩소-비용 가설 직접 정량화).

korea_ensemble.py 와 동일 basket_ensemble/overlay 수식 (parity). 라이브 무변경 — 측정만.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_gate_hysteresis.py
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
BAND = 0.01                      # 히스테리시스 ±1% (a-priori, 튜닝 아님)
TRANS_COSTS = [0.0, 0.002, 0.004]  # 게이트 전환당 extra cost 민감도
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


def basket_ensemble(close, dvol) -> pd.Series:
    """korea_ensemble.basket_ensemble 와 동일 (gate-무관 momentum basket, 비용 차감)."""
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
        mean_past = df.mean(axis=1)
        blended = df.rank().mean(axis=1)
        cand = blended[mean_past > 0]
        if len(cand) == 0:
            out[dates[i]] = 0.0; continue
        picks = cand.nlargest(min(N_POS, len(cand))).index
        out[dates[i]] = float(np.mean([pos_ret(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t])
                                       for t in picks])) - COST
    return pd.Series(out).sort_index()


def stances(b, index, mode: str):
    """rebalance 날짜별 게이트 stance(0/1) + idx/sma ratio. mode: 'binary' | 'hyst'."""
    sma = index.rolling(200, min_periods=100).mean()
    st, ratios = [], []
    prev = 0  # 초기 stance = 현금 (보수적: 데이터 부족 시 OFF)
    for d in b.index:
        iv, sv = index.asof(d), sma.asof(d)
        if pd.isna(iv) or pd.isna(sv) or sv <= 0:
            cur = 0; ratios.append(np.nan)
        else:
            r = iv / sv; ratios.append(r)
            if mode == "binary":
                cur = 1 if r > 1.0 else 0
            else:  # hysteresis
                if r > 1.0 + BAND:
                    cur = 1
                elif r < 1.0 - BAND:
                    cur = 0
                else:
                    cur = prev      # band 내 = 이전 stance 유지
        st.append(cur); prev = cur
    return pd.Series(st, index=b.index), pd.Series(ratios, index=b.index)


def overlay_with_stance(b, stance, trans_cost=0.0):
    """stance(0/1) × vol-target × basket. trans_cost: 게이트 전환당 extra 차감."""
    tgt = TARGET_VOL / np.sqrt(RPY); vals = b.values; rets = []
    sv = stance.values; prev = 0
    for k, (d, x) in enumerate(b.items()):
        exp = float(sv[k])
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, CAP)) if rv > 1e-9 else CAP
        r = exp * x
        if trans_cost and sv[k] != prev:   # 전환 발생 → extra cost
            r -= trans_cost
        rets.append(r); prev = sv[k]
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


def sub_ret(s, ps, pe):
    seg = s[(s.index >= ps) & (s.index <= pe)]
    return float(np.prod(1 + seg.values) - 1) if len(seg) else 0.0


def n_flips(stance):
    sv = stance.values
    return int(np.sum(sv[1:] != sv[:-1]))


def main() -> int:
    panel = pd.read_parquet(REPORTS / CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    index = fdr.DataReader(ICODE, DATA_START, END)["Close"]
    index.index = pd.to_datetime(index.index).normalize()

    b = basket_ensemble(close, dvol)
    n_rebal = len(b)

    st_bin, ratios = stances(b, index, "binary")
    st_hyst, _ = stances(b, index, "hyst")

    # 1. 플립 빈도
    flips_bin, flips_hyst = n_flips(st_bin), n_flips(st_hyst)
    in_band = (ratios.sub(1.0).abs() < BAND)
    n_in_band = int(in_band.sum())
    # danger band 안에서 발생한 binary 플립
    flip_mask = pd.Series(False, index=b.index)
    sv = st_bin.values
    flip_mask.iloc[1:] = sv[1:] != sv[:-1]
    flips_in_band = int((flip_mask & in_band).sum())

    logger.info("=" * 84)
    logger.info(f"KOSDAQ 게이트 휩소 측정 — full-cycle {b.index.min():%Y-%m} ~ {b.index.max():%Y-%m}, "
                f"리밸런스 {n_rebal}회 (~월 1회)")
    logger.info("=" * 84)
    logger.info(f"binary 게이트 stance 전환: {flips_bin}회 / {n_rebal} 리밸런스 "
                f"= {flips_bin / n_rebal:.1%}  (전환당 ~{n_rebal / max(flips_bin,1) * HOLD / 252:.1f}년)")
    logger.info(f"히스테리시스(±{BAND:.0%}) 전환: {flips_hyst}회 ({flips_hyst - flips_bin:+d} vs binary)")
    logger.info(f"danger band(|idx/sma-1|<{BAND:.0%}) 안에 있던 리밸런스: {n_in_band}/{n_rebal} = "
                f"{n_in_band / n_rebal:.1%}, 그중 binary 플립 발생: {flips_in_band}회")
    logger.info("")

    # 2~4. 비용 민감도별 binary vs hysteresis
    logger.info("전환 추가비용 민감도 (게이트 전환당 extra cost) — full-cycle:")
    logger.info(f"  {'trans_cost':>10} | {'binary Sharpe/ann/MDD':>30} | {'hyst Sharpe/ann/MDD':>30}")
    out_costs = {}
    for tc in TRANS_COSTS:
        rb = overlay_with_stance(b, st_bin, tc)
        rh = overlay_with_stance(b, st_hyst, tc)
        sb, sh = stat(rb.values), stat(rh.values)
        out_costs[tc] = {"binary": sb, "hyst": sh}
        logger.info(f"  {tc:>10.2%} | {sb['sharpe']:+.2f} / {sb['annual']:+.1%} / {sb['mdd']:.1%}"
                    f"        | {sh['sharpe']:+.2f} / {sh['annual']:+.1%} / {sh['mdd']:.1%}")
    logger.info("")

    # 3. crash window 참여 (trans_cost=0)
    rb0 = overlay_with_stance(b, st_bin, 0.0)
    rh0 = overlay_with_stance(b, st_hyst, 0.0)
    logger.info("crash window 수익 (binary vs hyst) — 상단버퍼 재진입 지연 검증:")
    crash_out = {}
    for c, (ps, pe) in CRASHES.items():
        cb, ch = sub_ret(rb0, ps, pe), sub_ret(rh0, ps, pe)
        crash_out[c] = {"binary": cb, "hyst": ch}
        flag = "  ← hyst 손해" if ch < cb - 0.005 else ("  ← hyst 이득" if ch > cb + 0.005 else "")
        logger.info(f"  {c:<12}: binary={cb:+.1%}  hyst={ch:+.1%}{flag}")
    logger.info("")

    # VERDICT
    base = out_costs[0.004]  # 보수적 전환비용 0.4% 가정
    d_sh = base["hyst"]["sharpe"] - base["binary"]["sharpe"]
    covid = crash_out["2020 COVID"]
    covid_hurt = covid["hyst"] < covid["binary"] - 0.005
    logger.info("VERDICT:")
    logger.info(f"  · 게이트 전환 빈도 = {flips_bin}/{n_rebal} ({flips_bin/n_rebal:.0%}) → "
                f"{'드묾(휩소 가설 약함)' if flips_bin/n_rebal < 0.25 else '잦음(휩소 가설 지지)'}")
    logger.info(f"  · 전환비용 0.4% 가정에서도 hyst−binary Sharpe = {d_sh:+.2f} → "
                f"{'히스테리시스 이득' if d_sh > 0.03 else ('무차이' if abs(d_sh) <= 0.03 else '히스테리시스 손해')}")
    logger.info(f"  · COVID 참여: {'hyst가 깎음(상단버퍼 약점 실증)' if covid_hurt else 'hyst 무해'}")
    accept = (flips_bin / n_rebal >= 0.25) and (d_sh > 0.03) and (not covid_hurt)
    logger.info(f"  → 채택? {'YES (히스테리시스 정당화됨)' if accept else 'NO (불필요/유해 — 라이브 무변경)'}")
    logger.info("=" * 84)

    p = REPORTS / "korea_gate_hysteresis.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_rebalances": n_rebal, "band": BAND,
        "flips_binary": flips_bin, "flips_hyst": flips_hyst,
        "n_in_band": n_in_band, "flips_in_band": flips_in_band,
        "cost_sensitivity": {f"{k:.3f}": v for k, v in out_costs.items()},
        "crash_participation": crash_out, "accept": accept,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
