"""KOSPI value/quality factor PIT 백테스트 (V4, 2026-05-30).

DART 펀더멘털(dart_fetch.py 캐시) + KOSPI 가격(long cache) → PIT factor 백테스트.

PIT lag (look-ahead 금지 핵심): 연도 Y 사업보고서는 Y+1 ~3월 공시(90일 마감) →
  - rebalance date D 의 month>=4 면 usable annual = D.year-1
  - month<4(1~3월)면 usable = D.year-2
  → 그 시점 실제로 알 수 있던 가장 최근 연간만 사용.

factor (long top-N, hold 20d, long-only, 상폐손실 반영):
  - quality (순수 DART, 완전 survivorship-free): ROE=순익/자본, 부채비율=부채/자본(↓),
    순마진=순익/매출
  - value (시총=close×현재주식수, live-only=survivor-biased, 낙관적 플래그):
    PBR=시총/자본(↓), earnings_yield=순익/시총(↑)
  - composite: z(EY)+z(ROE)-z(debt) (value+quality)

raw + gate(KS11 200d SMA). full-cycle Sharpe 0.5+ 나오면 KOSPI 엔진 후보.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/kospi_fundamental_factor.py
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

HOLD, LIQ_TOP, N_POS = 20, 100, 20
COST, DELIST_PEN = 0.004, 0.5
REPORTS = Path("v3/research/reports")
PRICE_CACHE = REPORTS / "korea_kospi_long_cache.parquet"
FUND_CACHE = REPORTS / "kospi_fundamentals.parquet"
RPY = 252 / HOLD


def usable_year(d: pd.Timestamp) -> int:
    """PIT: D 시점 알 수 있던 가장 최근 연간 사업보고서 연도."""
    return d.year - 1 if d.month >= 4 else d.year - 2


def pos_ret(path, entry):
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    return (path[lv] / entry - 1.0 - DELIST_PEN) if lv is not None and path[lv] > 0 else -1.0


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def compute_factors(tickers, close_row, shares, fund_by_ty, uy):
    """시점별 factor DataFrame. fund_by_ty[(ticker,year)]=dict. uy=usable year."""
    recs = {}
    for t in tickers:
        # 가장 최근 year <= uy
        fy = None
        for y in (uy, uy - 1):
            if (t, y) in fund_by_ty:
                fy = fund_by_ty[(t, y)]; break
        if fy is None or not fy.get("equity") or fy["equity"] <= 0:
            continue
        eq, debt, ni = fy["equity"], fy.get("debt"), fy.get("net_income")
        rev = fy.get("revenue")
        px = close_row.get(t)
        mcap = px * shares[t] if (px and t in shares and shares[t] > 0) else None
        rec = {
            "roe": (ni / eq) if ni is not None else np.nan,
            "debt_ratio": (debt / eq) if debt is not None else np.nan,
            "net_margin": (ni / rev) if (ni is not None and rev and rev > 0) else np.nan,
            "pbr": (mcap / eq) if mcap else np.nan,
            "earnings_yield": (ni / mcap) if (mcap and ni is not None) else np.nan,
        }
        recs[t] = rec
    return pd.DataFrame(recs).T


def pick(fdf: pd.DataFrame, signal: str) -> list:
    if fdf.empty:
        return []
    if signal == "value_pbr":
        s = fdf["pbr"].dropna(); s = s[s > 0]; return list(s.nsmallest(N_POS).index)
    if signal == "value_ey":
        s = fdf["earnings_yield"].dropna(); return list(s.nlargest(N_POS).index)
    if signal == "quality_roe":
        s = fdf["roe"].dropna(); return list(s.nlargest(N_POS).index)
    if signal == "quality_lowdebt":
        s = fdf["debt_ratio"].dropna(); s = s[s >= 0]; return list(s.nsmallest(N_POS).index)
    if signal == "composite":
        d = fdf.dropna(subset=["earnings_yield", "roe", "debt_ratio"])
        if len(d) < N_POS:
            return []
        score = zscore(d["earnings_yield"]) + zscore(d["roe"]) - zscore(d["debt_ratio"])
        return list(score.nlargest(N_POS).index)
    return []


def backtest(close, dvol, shares, fund_by_ty, signal, gate=None):
    dates = close.index.tolist(); rets, prev = [], set()
    for i in range(60, len(dates) - HOLD, HOLD):
        d = dates[i]
        if gate is not None and gate.asof(d) != True:
            rets.append(0.0); prev = set(); continue
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            rets.append(0.0); prev = set(); continue
        pool = list(liq.nlargest(LIQ_TOP).index)
        fdf = compute_factors(pool, close.iloc[i], shares, fund_by_ty, usable_year(d))
        picks = pick(fdf, signal)
        if not picks:
            rets.append(0.0); prev = set(); continue
        pr = [pos_ret(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t]) for t in picks]
        turn = len(set(picks) - prev) / max(len(picks), 1)
        rets.append(float(np.mean(pr)) - COST * turn); prev = set(picks)
    r = np.array(rets)
    if len(r) == 0 or not np.all(np.isfinite(r)):
        return {"annual": 0, "sharpe": 0, "mdd": 0, "n": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "win": float((r > 0).mean()), "n": len(r)}


def main() -> int:
    if not FUND_CACHE.exists():
        logger.error(f"펀더멘털 캐시 없음: {FUND_CACHE} — dart_fetch.py 먼저 (백그라운드 완료 대기)")
        return 1
    panel = pd.read_parquet(PRICE_CACHE)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    fund = pd.read_parquet(FUND_CACHE)
    fund_by_ty = {(r.ticker, int(r.year)): {"equity": r.equity, "debt": r.debt,
                  "net_income": r.net_income, "revenue": r.revenue}
                  for r in fund.itertuples()}
    logger.info(f"price {close.shape[1]}종목, fundamentals {fund['ticker'].nunique()}종목 "
                f"{int(fund['year'].min())}~{int(fund['year'].max())}")

    # 현재 주식수 (value 시총용 — live only, survivor bias 플래그)
    listing = fdr.StockListing("KOSPI")
    shares = {str(r.Code).zfill(6): float(r.Stocks) for r in listing.itertuples()
              if pd.notna(getattr(r, "Stocks", None))}
    ks = fdr.DataReader("KS11", "2014-01-01", "2026-05-01")["Close"]; ks.index = pd.to_datetime(ks.index).normalize()
    gate = ks > ks.rolling(200, min_periods=100).mean()

    logger.info("=" * 76)
    logger.info("KOSPI 펀더멘털 factor — PIT (raw / +gate)")
    logger.info("=" * 76)
    out = {}
    for sig in ["value_pbr", "value_ey", "quality_roe", "quality_lowdebt", "composite"]:
        raw = backtest(close, dvol, shares, fund_by_ty, sig)
        g = backtest(close, dvol, shares, fund_by_ty, sig, gate=gate)
        out[sig] = {"raw": raw, "gate": g}
        bias = " (survivor-biased)" if sig.startswith("value") or sig == "composite" else " (clean SF)"
        logger.info(f"  {sig:16s}{bias}")
        logger.info(f"      raw : annual={raw['annual']:+.1%}  Sharpe={raw['sharpe']:+.2f}  MDD={raw['mdd']:.1%}  n={raw['n']}")
        logger.info(f"      +gate: annual={g['annual']:+.1%}  Sharpe={g['sharpe']:+.2f}  MDD={g['mdd']:.1%}")

    flat = [(f"{k}_{m}", v[m]) for k, v in out.items() for m in ("raw", "gate")]
    best = max(flat, key=lambda kv: kv[1]["sharpe"])
    logger.info("\nVERDICT:")
    logger.info(f"  best: {best[0]} Sharpe={best[1]['sharpe']:.2f} annual={best[1]['annual']:+.1%}")
    clean = max(((k, v["raw"]) for k, v in out.items() if k.startswith("quality")),
                key=lambda kv: kv[1]["sharpe"])
    logger.info(f"  best CLEAN(quality, survivorship-free): {clean[0]} Sharpe={clean[1]['sharpe']:.2f}")
    logger.info("=" * 76)

    p = REPORTS / "kospi_fundamental_factor.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "results": out}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
