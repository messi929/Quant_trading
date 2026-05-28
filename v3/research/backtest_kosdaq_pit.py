"""KOSDAQ point-in-time universe 백테스트 (V4, 2026-05-29).

look-ahead 제거: 이전 백테스트는 "현재(2026) 시총 상위 150"을 universe로 써서
사후 승자(폭등 종목) 편향이 있었음. 진짜는 각 rebalance date에 "그 시점
거래대금(Close×Volume) 상위 M"을 동적 universe로 (현재 시총 무관).

전략 (이전과 동일, universe만 point-in-time):
  - 후보 pool: 현재 KOSDAQ 시총 상위 400 + 2021~ 상폐 250 (넓게 + 상폐 포함)
  - 각 rebalance(20d): 그 시점 거래대금 상위 M → past60>0 momentum 상위 N
  - equal weight, 20일 보유, 상폐 손실(-50%) + 비용(0.4%) 반영

Usage:
    PYTHONPATH=. python v3/research/backtest_kosdaq_pit.py
"""

from __future__ import annotations

import sys, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import warnings; warnings.filterwarnings("ignore")

import argparse
import FinanceDataReader as fdr

LOOKBACK = 60
HOLD = 20
N_CAND = 400          # 현재 시총 상위 후보 (넓게)
LIQ_TOP = 100         # 각 시점 거래대금 상위 (point-in-time universe)
START = "2021-01-01"
END = "2026-05-01"
REBAL_PER_YEAR = 252 / HOLD

MARKET = "KOSDAQ"     # set by argparse
CACHE = Path("v3/research/reports/korea_kosdaq_pit_cache.parquet")


def build_cache() -> pd.DataFrame:
    if CACHE.exists():
        logger.info(f"Loading cache: {CACHE}")
        return pd.read_parquet(CACHE)
    logger.info(f"Building {MARKET} universe (시총상위 400 + 상폐 250)...")
    kq = fdr.StockListing(MARKET)
    kq["Marcap"] = pd.to_numeric(kq["Marcap"], errors="coerce")
    live = kq.nlargest(N_CAND, "Marcap")["Code"].tolist()
    delist = fdr.StockListing("KRX-DELISTING")
    delist["DelistingDate"] = pd.to_datetime(delist["DelistingDate"], errors="coerce")
    deln = delist[(delist["Market"] == MARKET) & (delist["DelistingDate"] >= "2021-01-01")]["Symbol"].tolist()
    deln = [c for c in deln if isinstance(c, str) and c.isdigit() and len(c) == 6][:250]
    codes = live + deln
    logger.info(f"  candidates: {len(codes)}")
    frames = []
    for i, c in enumerate(codes, 1):
        try:
            df = fdr.DataReader(c, START, END)
            if df is None or df.empty or len(df) < 80:
                continue
            s = df[["Close", "Volume"]].copy()
            s.columns = ["close", "volume"]
            s["date"] = pd.to_datetime(s.index).normalize()
            s["ticker"] = c
            frames.append(s.reset_index(drop=True))
        except Exception:
            pass
        if i % 50 == 0:
            logger.info(f"  {i}/{len(codes)} (ok={len(frames)})")
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(CACHE)
    logger.info(f"  cached {len(panel)} rows, {panel['ticker'].nunique()} tickers")
    return panel


def backtest(close: pd.DataFrame, amount: pd.DataFrame, n_pos: int,
             cost: float = 0.004, delist_pen: float = 0.5) -> dict:
    dates = close.index.tolist()
    rets = []
    prev = set()
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        # point-in-time 거래대금 상위 universe
        liq = amount.iloc[i].dropna()
        if len(liq) < 20:
            rets.append(0.0); prev = set(); continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - LOOKBACK][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            rets.append(0.0); prev = set(); continue
        picks = trend.nlargest(min(n_pos, len(trend))).index
        pr = []
        for t in picks:
            e = close.iloc[i][t]; x = close.iloc[i + HOLD][t]
            if pd.notna(x):
                pr.append(x / e - 1.0)
            else:
                w = close.iloc[i:i + HOLD + 1][t]
                lv = w.last_valid_index()
                pr.append((w[lv] / e - 1.0 - delist_pen) if lv is not None and w[lv] > 0 else -1.0)
        gross = float(np.mean(pr))
        turnover = len(set(picks) - prev) / max(len(picks), 1)
        rets.append(gross - cost * turnover)
        prev = set(picks)
    rets = np.array(rets)
    if len(rets) == 0:
        return {}
    eq = np.cumprod(1 + rets)
    total = float(eq[-1] - 1)
    ann = float((1 + total) ** (REBAL_PER_YEAR / len(rets)) - 1)
    vol = float(rets.std() * np.sqrt(REBAL_PER_YEAR))
    sharpe = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return {"n_pos": n_pos, "total_return": total, "annual_return": ann,
            "sharpe": sharpe, "mdd": mdd, "win_rate": float((rets > 0).mean()),
            "n_rebal": len(rets)}


def main() -> int:
    global MARKET, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KOSDAQ", choices=["KOSDAQ", "KOSPI"])
    args = ap.parse_args()
    MARKET = args.market
    CACHE = Path(f"v3/research/reports/korea_{MARKET.lower()}_pit_cache.parquet")
    panel = build_cache()
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    amount = close * vol   # 거래대금
    logger.info(f"universe pool: {close.shape[1]} tickers, {close.shape[0]} days")

    idx = fdr.DataReader("KQ11", START, END)["Close"]
    idx_ann = (idx.iloc[-1] / idx.iloc[0]) ** (365 / (idx.index[-1] - idx.index[0]).days) - 1
    logger.info(f"KOSDAQ 지수 annual: {idx_ann:+.1%}")

    logger.info("=" * 70)
    logger.info("POINT-IN-TIME universe backtest (각 시점 거래대금 상위 → momentum)")
    logger.info(f"  거래대금 상위 {LIQ_TOP} pool, 상폐손실 50%, 비용 0.4%")
    logger.info("=" * 70)
    results = []
    for n in [10, 20, 30]:
        r = backtest(close, amount, n)
        results.append(r)
        logger.info(f"  N={n:2d}: annual={r['annual_return']:+.1%}  Sharpe={r['sharpe']:+.2f}  "
                    f"MDD={r['mdd']:.1%}  win={r['win_rate']:.0%}  (rebal={r['n_rebal']})")
    best = results[1]  # N=20
    logger.info("")
    logger.info("VERDICT (point-in-time, N=20):")
    if best["annual_return"] > 0.10 and best["sharpe"] > 0.7:
        logger.info(f"  PASS — net annual {best['annual_return']:+.1%}, Sharpe {best['sharpe']:.2f}. 진짜 edge.")
    elif best["annual_return"] > 0.05:
        logger.info(f"  MARGINAL — annual {best['annual_return']:+.1%} (무위험 근처).")
    else:
        logger.info(f"  FAIL — annual {best['annual_return']:+.1%}. look-ahead 제거 시 소멸.")
    logger.info(f"  (이전 look-ahead universe: +57.2%)")
    logger.info("=" * 70)

    out = Path(f"v3/research/reports/{MARKET.lower()}_pit_backtest.json")
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_tickers": int(close.shape[1]), "liq_top": LIQ_TOP,
        "kosdaq_index_annual": float(idx_ann), "grid": results,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
