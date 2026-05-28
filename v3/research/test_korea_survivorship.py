"""한국 momentum survivorship 간접 검증 (V4, 2026-05-29).

문제: yfinance는 상폐 종목 데이터를 안 줌 (오스템임플란트 048260 EMPTY).
→ 현재 살아있는 종목만 봄 = survivorship bias. KOSDAQ momentum +0.040~0.063이
   "진짜 edge"인지 "살아남은 것만 본 착시"인지 갈라야 함.

완전한 survivorship-free는 유료 데이터(KRX/FnGuide) 필요. 무료 범위 간접 검증:

  1. Sub-period 분할 (2022-24 vs 2024-26): momentum 일관성.
     survivorship bias 증상 = 최근 sub-period에 과대 (생존자가 최근 더 많이 살아있음).
     모든 sub-period 일관 양수면 → bias보다 진짜 신호 가능성.
  2. TS vs CS 비교: TS momentum(각 종목 자기 시계열)은 CS(종목 간 비교)보다
     survivorship에 덜 민감. TS도 양수면 더 robust.
  3. Drawdown 경험 종목 포함 효과: 큰 하락(-40%+) 겪고 생존한 종목 비율.

데이터: yfinance 한국 종목 (전체 universe 아닌 샘플 — 간접 검증 목적).

Usage:
    PYTHONPATH=. python v3/research/test_korea_survivorship.py
"""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

# 한국 universe (KOSPI + KOSDAQ 샘플, momentum 검증과 동일 + 확대)
KOREA = [
    # KOSPI
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
    "000270.KS", "005490.KS", "035420.KS", "035720.KS", "051910.KS",
    "006400.KS", "028260.KS", "105560.KS", "055550.KS", "012330.KS",
    "066570.KS", "003670.KS", "015760.KS", "017670.KS", "034730.KS",
    "009150.KS", "011200.KS", "086790.KS", "316140.KS", "024110.KS",
    # KOSDAQ
    "086520.KQ", "196170.KQ", "247540.KQ", "028300.KQ", "066970.KQ",
    "357780.KQ", "058470.KQ", "240810.KQ", "098460.KQ", "022100.KQ",
    "263750.KQ", "293490.KQ", "095340.KQ", "041510.KQ", "067310.KQ",
    "078600.KQ", "086900.KQ", "214150.KQ", "112040.KQ", "039030.KQ",
    "145020.KQ", "278280.KQ", "036930.KQ", "140860.KQ", "200670.KQ",
]

LOOKBACK = 60
HORIZON = 20


def fetch(ticker, start, end):
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 150:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()[["Date", "Close"]].copy()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = ticker
        return df
    except Exception:
        return None


def momentum(panel, lookback, horizon):
    from scipy.stats import spearmanr
    piv = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dates = piv.index.tolist()
    ts_up, ts_down, cs = [], [], []
    for i in range(lookback, len(dates) - horizon, horizon):
        past = (piv.iloc[i] / piv.iloc[i - lookback] - 1.0)
        fwd = (piv.iloc[i + horizon] / piv.iloc[i] - 1.0)
        v = past.notna() & fwd.notna()
        past, fwd = past[v], fwd[v]
        if len(past) < 5:
            continue
        ts_up.extend(fwd[past > 0].tolist())
        ts_down.extend(fwd[past < 0].tolist())
        if past.std() > 1e-9 and fwd.std() > 1e-9:
            rho, _ = spearmanr(past.to_numpy(), fwd.to_numpy())
            if np.isfinite(rho):
                cs.append(float(rho))
    return {
        "ts_premium": (np.mean(ts_up) - np.mean(ts_down)) if ts_up and ts_down else 0.0,
        "ts_long": float(np.mean(ts_up)) if ts_up else 0.0,
        "cs_ic": float(np.mean(cs)) if cs else 0.0,
        "n_long": len(ts_up),
    }


def main() -> int:
    logger.info(f"Fetching {len(KOREA)} Korea tickers (2021-2026)...")
    frames, kept = [], []
    for t in KOREA:
        df = fetch(t, "2021-01-01", "2026-05-01")
        if df is not None:
            frames.append(df)
            kept.append(t)
    panel = pd.concat(frames, ignore_index=True)
    logger.info(f"  {len(kept)}/{len(KOREA)} tickers")

    # ── 1. Drawdown 경험 종목 비율 (survivor 특성) ──
    piv = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dd_count = 0
    for t in piv.columns:
        s = piv[t].dropna()
        if len(s) < 100:
            continue
        roll_max = s.cummax()
        max_dd = ((s - roll_max) / roll_max).min()
        if max_dd < -0.40:
            dd_count += 1
    logger.info(f"  -40%+ drawdown 경험 종목: {dd_count}/{len(kept)} "
                f"({dd_count/len(kept):.0%}) — survivor가 큰 하락 후 회복했는지 지표")

    # ── 2. Sub-period 분할 ──
    periods = {
        "full_2021_2026": ("2021-01-01", "2026-05-01"),
        "sub1_2021_2023": ("2021-01-01", "2023-05-01"),
        "sub2_2023_2026": ("2023-05-01", "2026-05-01"),
        "recent_2024_2026": ("2024-05-01", "2026-05-01"),
    }
    logger.info("=" * 64)
    logger.info("SUB-PERIOD momentum (lookback 60d, hold 20d)")
    logger.info("survivorship 증상 = 최근 sub-period에 과대. 일관되면 robust.")
    logger.info("=" * 64)
    report = {"dd40_ratio": dd_count / len(kept), "n_tickers": len(kept), "periods": {}}
    for pname, (ps, pe) in periods.items():
        sub = panel[(panel["date"] >= ps) & (panel["date"] < pe)]
        m = momentum(sub, LOOKBACK, HORIZON)
        report["periods"][pname] = m
        logger.info(
            f"  {pname:20s}: ts_premium={m['ts_premium']:+.4f}  "
            f"ts_long={m['ts_long']:+.4f}  cs_ic={m['cs_ic']:+.4f}  (n={m['n_long']})"
        )

    # ── 판정 ──
    logger.info("=" * 64)
    subs = [report["periods"][p] for p in ["sub1_2021_2023", "sub2_2023_2026"]]
    cs_consistent = all(s["cs_ic"] > 0 for s in subs)
    ts_consistent = all(s["ts_long"] > 0 for s in subs)
    logger.info("VERDICT:")
    if cs_consistent and ts_consistent:
        logger.info("  PASS(간접) — 모든 sub-period에서 momentum 양수 일관. survivorship보다")
        logger.info("    진짜 신호 가능성. 단 완전 검증은 delisted 포함 유료 데이터 필요.")
    elif report["periods"]["recent_2024_2026"]["cs_ic"] > 2 * report["periods"]["sub1_2021_2023"]["cs_ic"]:
        logger.info("  WARN — 최근 편중 (survivorship bias 의심). 신중.")
    else:
        logger.info("  MIXED — sub-period 일관성 부분적. 추가 검증 필요.")
    logger.info("=" * 64)

    out = Path("v3/research/reports/korea_survivorship.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **report,
        "note": "yfinance 상폐 종목 데이터 없음 → 간접 검증만. 완전 검증은 유료 데이터.",
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
