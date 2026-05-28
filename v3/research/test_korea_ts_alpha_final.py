"""한국 TS momentum beta/alpha 최종 확정 (V4, 2026-05-29).

마지막 게이트: ts_long +2.78%(시장초과, 전체기간)가 진짜 market-neutral alpha인지,
상승장 long-only beta인지. 2022 하락기 포함 sub-period로 확정.

방법:
  각 rebalance date d:
    추세종목(past 60d > 0) 20d forward 평균
    − KOSDAQ 지수(KQ11) 같은 20d forward
    = market-neutral momentum alpha (그 시점)
  sub-period(2021-22 하락 / 2022-23 / 2023-24 / 2024-26)별 평균.
  하락기 포함 모든 sub-period 양수면 → 진짜 alpha → 시스템 설계 진행.
  하락기 음수/0이면 → 상승장 beta 거품.

universe: KOSDAQ survivorship-free (live top150 + 2021~ 상폐). OHLCV는
parquet 캐시 (재수집 방지).

Usage:
    PYTHONPATH=. python v3/research/test_korea_ts_alpha_final.py
"""

from __future__ import annotations

import sys, json, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr

LOOKBACK = 60
HORIZON = 20
N_LIVE = 150
START = "2021-01-01"
END = "2026-05-01"
CACHE = Path("v3/research/reports/korea_kosdaq_ohlcv_cache.parquet")


def build_universe_ohlcv() -> pd.DataFrame:
    if CACHE.exists():
        logger.info(f"Loading OHLCV cache: {CACHE}")
        return pd.read_parquet(CACHE)

    logger.info("Building KOSDAQ survivorship-free universe...")
    kq = fdr.StockListing("KOSDAQ")
    kq["Marcap"] = pd.to_numeric(kq["Marcap"], errors="coerce")
    live = kq.nlargest(N_LIVE, "Marcap")["Code"].tolist()
    delist = fdr.StockListing("KRX-DELISTING")
    delist["DelistingDate"] = pd.to_datetime(delist["DelistingDate"], errors="coerce")
    deln = delist[(delist["Market"] == "KOSDAQ") & (delist["DelistingDate"] >= "2021-01-01")]["Symbol"].tolist()
    deln = [c for c in deln if isinstance(c, str) and c.isdigit() and len(c) == 6]
    codes = live + deln
    logger.info(f"  live {len(live)} + delisted {len(deln)} = {len(codes)}")

    frames = []
    for i, c in enumerate(codes, 1):
        try:
            df = fdr.DataReader(c, START, END)
            if df is None or df.empty or len(df) < 80:
                continue
            s = df[["Close"]].copy()
            s.columns = ["close"]
            s["date"] = pd.to_datetime(s.index).normalize()
            s["ticker"] = c
            frames.append(s.reset_index(drop=True))
        except Exception:
            pass
        if i % 50 == 0:
            logger.info(f"  {i}/{len(codes)}")
    panel = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE)
    logger.info(f"  cached {len(panel)} rows → {CACHE}")
    return panel


def main() -> int:
    panel = build_universe_ohlcv()
    piv = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    n_tickers = piv.shape[1]
    logger.info(f"universe pivot: {piv.shape}")

    # KOSDAQ 지수
    idx = fdr.DataReader("KQ11", START, END)["Close"]
    idx.index = pd.to_datetime(idx.index).normalize()

    dates = piv.index.tolist()

    # 각 rebalance date: 추세종목 fwd − 지수 fwd = market-neutral alpha
    records = []
    for i in range(LOOKBACK, len(dates) - HORIZON, HORIZON):
        d = dates[i]
        d_fwd = dates[i + HORIZON]
        past = (piv.iloc[i] / piv.iloc[i - LOOKBACK] - 1.0)
        fwd = (piv.iloc[i + HORIZON] / piv.iloc[i] - 1.0)
        v = past.notna() & fwd.notna()
        past, fwd = past[v], fwd[v]
        trend = fwd[past > 0]
        if len(trend) < 5:
            continue
        # 지수 fwd
        try:
            iv0 = idx.asof(d)
            iv1 = idx.asof(d_fwd)
            idx_fwd = iv1 / iv0 - 1.0 if iv0 and iv0 > 0 else np.nan
        except Exception:
            idx_fwd = np.nan
        if not np.isfinite(idx_fwd):
            continue
        records.append({
            "date": d,
            "trend_fwd": float(trend.mean()),
            "idx_fwd": float(idx_fwd),
            "alpha": float(trend.mean() - idx_fwd),
            "n_trend": int(len(trend)),
        })

    rec = pd.DataFrame(records)
    logger.info(f"rebalance points: {len(rec)}")

    # sub-period
    periods = {
        "2021-2022(하락포함)": ("2021-01-01", "2022-12-31"),
        "2022(하락기)": ("2022-01-01", "2022-12-31"),
        "2023-2024": ("2023-01-01", "2024-12-31"),
        "2024-2026(최근)": ("2024-01-01", "2026-05-01"),
        "full": ("2021-01-01", "2026-05-01"),
    }
    logger.info("=" * 64)
    logger.info("한국 TS momentum market-neutral ALPHA by sub-period")
    logger.info("alpha = 추세종목(past60>0) 20d fwd − KOSDAQ지수 20d fwd")
    logger.info("=" * 64)
    out_periods = {}
    for pname, (ps, pe) in periods.items():
        sub = rec[(rec["date"] >= ps) & (rec["date"] <= pe)]
        if len(sub) == 0:
            continue
        a = float(sub["alpha"].mean())
        tf = float(sub["trend_fwd"].mean())
        if_ = float(sub["idx_fwd"].mean())
        out_periods[pname] = {"alpha": a, "trend_fwd": tf, "idx_fwd": if_, "n": len(sub)}
        logger.info(f"  {pname:18s}: alpha={a:+.4f}  (추세={tf:+.4f}, 지수={if_:+.4f}, n={len(sub)})")

    logger.info("")
    logger.info("VERDICT:")
    down = out_periods.get("2022(하락기)", {}).get("alpha", None)
    full = out_periods.get("full", {}).get("alpha", 0)
    all_pos = all(v["alpha"] > 0 for k, v in out_periods.items() if k != "full")
    if all_pos and full > 0.01:
        logger.info(f"  PASS — 하락기 포함 모든 sub-period alpha 양수. 진짜 market-neutral")
        logger.info(f"    momentum alpha (full {full:+.4f}). 시스템 설계 진행 가능.")
    elif down is not None and down <= 0:
        logger.info(f"  FAIL — 2022 하락기 alpha={down:+.4f} ≤ 0. 상승장 beta 거품.")
    else:
        logger.info(f"  MIXED — sub-period 일관성 부분적. 신중.")
    logger.info("=" * 64)

    out = Path("v3/research/reports/korea_ts_alpha_final.json")
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_tickers": n_tickers, "lookback": LOOKBACK, "horizon": HORIZON,
        "periods": out_periods,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
