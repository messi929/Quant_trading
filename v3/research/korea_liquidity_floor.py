"""P2 — 종목별 절대 유동성 floor 측정 (2026-06-19).

동기: 사용자 #2 — 가뭄장에 상대 Top100 만으로는 100위 종목이 절대적으로 너무 얇아
작전성 테마주가 섞일 수 있음. 시장 타이밍(노출 컷)이 아니라 a-priori 한 *종목별
절대 거래대금 floor* 가 정당한가? 두 질문:

  (A) 체결성 sanity — 6/15 라이브 바스켓 20종목의 최근 거래대금이 1억 계좌 종목당
      ~500만 체결에 충분한가? (참여율 = 5M / 일거래대금).
  (B) full-cycle — pool 에 절대 floor(20d평균 거래대금 > X억) 추가 시 Sharpe/annual/
      MDD 손상되는가? floor 가 공짜면 tail 보호로 채택, 수익 깎으면 기각.

라이브 바스켓은 FDR 최근값, full-cycle 은 long 캐시. 라이브 무변경 — 측정만.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_liquidity_floor.py
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
RPY = 252 / HOLD
FILL_PER_NAME = 5_000_000          # 1억 / 20종목
FLOORS = [0, 5e8, 1e9, 2e9, 5e9]   # 종목별 20d평균 거래대금 절대 floor (원)
REPORTS = Path("v3/research/reports")
CACHE, ICODE = "korea_kosdaq_long_cache.parquet", "KQ11"
LIVE_BASKET = ["036930", "080220", "417840", "319660", "095610", "131290", "064290",
               "440110", "330860", "131970", "170920", "222800", "242040", "160980",
               "356680", "033160", "052710", "031330", "089970", "010170"]  # 6/15 picks(20)


def pos_ret(path, entry):
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    if lv is not None and path[lv] > 0:
        return path[lv] / entry - 1.0 - DELIST_PEN
    return -1.0


def basket_ensemble(close, dvol, floor=0.0) -> pd.Series:
    """floor: pool 진입 전 20d평균 거래대금 절대 하한 (원). 0 이면 baseline."""
    dvol_ma = dvol.rolling(20, min_periods=10).mean()
    dates = close.index.tolist(); out = {}
    for i in range(MAXLB, len(dates) - HOLD, HOLD):
        liq = dvol.iloc[i].dropna()
        if floor > 0:
            ma = dvol_ma.iloc[i]
            liq = liq[ma.reindex(liq.index) >= floor]
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


def overlay(b, gate):
    tgt = TARGET_VOL / np.sqrt(RPY); vals = b.values; rets = []
    for k, (d, x) in enumerate(b.items()):
        g = gate.asof(d); exp = 1.0 if (g is True or g == True) else 0.0  # noqa: E712 (numpy bool)
        if exp > 0 and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt / rv, 0, CAP)) if rv > 1e-9 else CAP
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


def live_basket_liquidity():
    """(A) 6/15 바스켓 20종목 최근 20d평균 거래대금 + 참여율."""
    logger.info("=" * 84)
    logger.info("(A) 6/15 라이브 바스켓 체결성 — 종목당 500만 체결 시 참여율")
    logger.info("=" * 84)
    rows = []
    for t in LIVE_BASKET:
        try:
            df = fdr.DataReader(t, "2026-05-01", "2026-06-18")
            adv = float((df["Close"] * df["Volume"]).tail(20).mean())  # 20d평균 거래대금(원)
            part = FILL_PER_NAME / adv if adv > 0 else np.nan
            rows.append((t, adv, part))
        except Exception as e:
            logger.warning(f"  {t}: fetch 실패 {e}")
            rows.append((t, np.nan, np.nan))
    rows.sort(key=lambda r: (r[1] if not np.isnan(r[1]) else 1e99))
    logger.info(f"  {'ticker':>8} | {'20d 거래대금(억)':>16} | {'참여율(5M/ADV)':>14}")
    for t, adv, part in rows:
        flag = "  ⚠️ 얇음" if (not np.isnan(adv) and adv < 1e9) else ""
        logger.info(f"  {t:>8} | {adv/1e8:>14.1f}억 | {part:>13.3%}{flag}")
    advs = [r[1] for r in rows if not np.isnan(r[1])]
    logger.info(f"  → 최저 거래대금 {min(advs)/1e8:.1f}억, 최대 참여율 {max(FILL_PER_NAME/a for a in advs):.2%}  "
                f"(<0.5%면 충격 무시 가능)")
    logger.info("")
    return {t: {"adv_krw": adv, "participation": part} for t, adv, part in rows}


def main() -> int:
    live = live_basket_liquidity()

    panel = pd.read_parquet(REPORTS / CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    index = fdr.DataReader(ICODE, "2014-01-01", "2026-05-01")["Close"]
    index.index = pd.to_datetime(index.index).normalize()
    gate = index > index.rolling(200, min_periods=100).mean()

    logger.info("=" * 84)
    logger.info("(B) full-cycle — 종목별 절대 거래대금 floor 민감도 (floor 가 수익 깎는가?)")
    logger.info("=" * 84)
    base = None; out_floors = {}
    logger.info(f"  {'floor':>8} | {'Sharpe':>7} | {'annual':>8} | {'MDD':>7} | {'vs baseline':>12}")
    for fl in FLOORS:
        s = overlay(basket_ensemble(close, dvol, floor=fl), gate)
        st = stat(s.values); out_floors[fl] = st
        if base is None:
            base = st
        d_sh = st["sharpe"] - base["sharpe"]
        lbl = "baseline" if fl == 0 else f"{fl/1e8:.0f}억"
        logger.info(f"  {lbl:>8} | {st['sharpe']:>+7.2f} | {st['annual']:>+8.1%} | {st['mdd']:>7.1%} | "
                    f"{d_sh:>+12.2f}")
    logger.info("")

    # VERDICT
    best_safe = max((fl for fl in FLOORS if out_floors[fl]["sharpe"] >= base["sharpe"] - 0.03),
                    default=0)
    advs = [v["adv_krw"] for v in live.values() if v["adv_krw"] and not np.isnan(v["adv_krw"])]
    thin = sum(1 for a in advs if a < 1e9)
    logger.info("VERDICT:")
    logger.info(f"  · 라이브 바스켓: 최저 거래대금 {min(advs)/1e8:.1f}억, 1억미만 종목 {thin}/{len(advs)}개. "
                f"종목당 5M 참여율 최대 {max(FILL_PER_NAME/a for a in advs):.2%} → "
                f"{'현 1억 규모 체결성 문제 없음' if max(FILL_PER_NAME/a for a in advs) < 0.005 else '체결성 주의'}")
    logger.info(f"  · full-cycle: Sharpe 무손상 최대 floor = {best_safe/1e8:.0f}억 "
                f"(0이면 floor 가 수익 깎음 → 기각)")
    if best_safe > 0:
        logger.info(f"  → {best_safe/1e8:.0f}억 floor 는 수익 무손상 + tail 보호 → 채택 고려 가능 "
                    f"(단 현 규모엔 미발동, 자본 스케일업 대비)")
    else:
        logger.info("  → 절대 floor 가 full-cycle 수익을 깎음 → 현 SPEC 유지, 기각")
    logger.info("=" * 84)

    p = REPORTS / "korea_liquidity_floor.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fill_per_name": FILL_PER_NAME, "floors": FLOORS,
        "live_basket": live, "full_cycle": {f"{k:.0f}": v for k, v in out_floors.items()},
        "best_safe_floor": best_safe,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
