"""한국 긴 history 다중 하락기 크래시 보험 검증 (V4, 2026-05-29).

robustness 점검의 최대 구멍: 표본 내 크래시 2022 단 1회. regime filter 의 크래시
보험 효과를 n=1 로 검증한 셈. 2014~ 로 확장해 다중 하락기로 n>1 검증:
  - 2018 Q4 (미중 무역분쟁 selloff)
  - 2020 Q1 (COVID 폭락, KOSPI -35%/KOSDAQ -40%)
  - 2022 (금리인상 약세장)
  - 2015-16 (KOSDAQ 조정)

핵심 테스트 = 절대수익 아님. 각 하락기에서 baseline(gate없음) vs regime gate 의
MDD/return 비교. universe 잔여 look-ahead bias 에 robust (gate 효과만 격리).

universe: 검증된 PIT 방법론 연장 — 현재 시총 top600 live + 2015~ 상폐 (wider pool),
2014-01~ OHLCV. 각 rebalance PIT 거래대금 top100 → momentum.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_long_history.py --build
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_long_history.py
"""

from __future__ import annotations

import argparse
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
N_LIVE = 600
DATA_START, DELIST_START, END = "2014-01-01", "2015-01-01", "2026-05-01"
RPY = 252 / HOLD
REPORTS = Path("v3/research/reports")
MARKETS = {"KOSDAQ": ("korea_kosdaq_long_cache.parquet", "KQ11"),
           "KOSPI": ("korea_kospi_long_cache.parquet", "KS11")}

CRASHES = {
    "2015-16 조정": ("2015-07-01", "2016-02-29"),
    "2018 Q4": ("2018-06-01", "2019-01-31"),
    "2020 COVID": ("2020-01-01", "2020-05-31"),
    "2022 약세장": ("2022-01-01", "2022-12-31"),
}


def build_cache(market: str, cache: Path):
    if cache.exists():
        return
    logger.info(f"[{market}] building long survivorship-free cache (top{N_LIVE} + 상폐 2015~)...")
    kq = fdr.StockListing(market)
    kq["Marcap"] = pd.to_numeric(kq["Marcap"], errors="coerce")
    live = kq.nlargest(N_LIVE, "Marcap")["Code"].tolist()
    dl = fdr.StockListing("KRX-DELISTING")
    dl["DelistingDate"] = pd.to_datetime(dl["DelistingDate"], errors="coerce")
    deln = dl[(dl["Market"] == market) & (dl["DelistingDate"] >= DELIST_START)]["Symbol"].tolist()
    deln = [c for c in deln if isinstance(c, str) and c.isdigit() and len(c) == 6]
    codes = list(dict.fromkeys(live + deln))
    logger.info(f"  live {len(live)} + delisted {len(deln)} = {len(codes)} candidates")
    frames = []
    for i, c in enumerate(codes, 1):
        try:
            df = fdr.DataReader(c, DATA_START, END)
            if df is None or df.empty or len(df) < 80:
                continue
            s = df[["Close", "Volume"]].copy(); s.columns = ["close", "volume"]
            s["date"] = pd.to_datetime(s.index).normalize(); s["ticker"] = c
            frames.append(s.reset_index(drop=True))
        except Exception:
            pass
        if i % 100 == 0:
            logger.info(f"  [{market}] {i}/{len(codes)} (ok={len(frames)})")
    panel = pd.concat(frames, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    logger.info(f"  [{market}] cached {len(panel)} rows, {panel['ticker'].nunique()} tickers")


def gate_series(index: pd.Series, kind: str, param: int) -> pd.Series:
    if kind == "none":
        return pd.Series(True, index=index.index)
    if kind == "sma":
        return index > index.rolling(param, min_periods=param // 2).mean()
    if kind == "mom":
        return (index / index.shift(param) - 1.0) > 0
    raise ValueError(kind)


def sleeve(close, dvol, gate: pd.Series) -> pd.Series:
    dates = close.index.tolist(); out = {}
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        d = dates[i]; g = gate.asof(d)
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
        return {"annual": 0, "sharpe": 0, "mdd": 0, "ret": 0, "n": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "ret": tot, "n": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    for m, (cfile, _) in MARKETS.items():
        build_cache(m, REPORTS / cfile)
    if args.build:
        logger.info("build 완료."); return 0

    out_all = {}
    for m, (cfile, icode) in MARKETS.items():
        panel = pd.read_parquet(REPORTS / cfile)
        close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
        dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
        index = fdr.DataReader(icode, DATA_START, END)["Close"]; index.index = pd.to_datetime(index.index).normalize()
        logger.info("=" * 80)
        logger.info(f"{m} long history: {close.shape[1]} tickers, {close.shape[0]} days "
                    f"({close.index[0].date()}~{close.index[-1].date()})")
        logger.info("=" * 80)

        base = sleeve(close, dvol, gate_series(index, "none", 0))
        g200 = sleeve(close, dvol, gate_series(index, "sma", 200))
        gm60 = sleeve(close, dvol, gate_series(index, "mom", 60))

        # 전체 기간
        logger.info("전체 기간 (gate없음 / sma200 / mom60):")
        for nm, s in [("none", base), ("sma200", g200), ("mom60", gm60)]:
            st = stat(s.values)
            logger.info(f"  {nm:7s}: annual={st['annual']:+.1%}  Sharpe={st['sharpe']:+.2f}  MDD={st['mdd']:.1%}  n={st['n']}")

        # 크래시별 (핵심): baseline vs gate MDD/return
        logger.info("크래시별 MDD / return  [base → sma200 → mom60]:")
        crash_out = {}
        for cname, (ps, pe) in CRASHES.items():
            rb = base[(base.index >= ps) & (base.index <= pe)].values
            r2 = g200[(g200.index >= ps) & (g200.index <= pe)].values
            r6 = gm60[(gm60.index >= ps) & (gm60.index <= pe)].values
            sb, s2, s6 = stat(rb), stat(r2), stat(r6)
            crash_out[cname] = {"base": sb, "sma200": s2, "mom60": s6}
            logger.info(f"  {cname:14s}: MDD {sb['mdd']:+.0%}→{s2['mdd']:+.0%}→{s6['mdd']:+.0%}  "
                        f"ret {sb['ret']:+.0%}→{s2['ret']:+.0%}→{s6['ret']:+.0%}  (n={sb['n']})")
        out_all[m] = {"full": {"none": stat(base.values), "sma200": stat(g200.values),
                               "mom60": stat(gm60.values)}, "crashes": crash_out}
        logger.info("")

    # 판정
    logger.info("VERDICT (크래시 보험 n>1 검증):")
    for m in MARKETS:
        wins = sum(1 for c in out_all[m]["crashes"].values()
                   if abs(c["sma200"]["mdd"]) < abs(c["base"]["mdd"]) - 0.02)
        tot = len(out_all[m]["crashes"])
        logger.info(f"  {m}: sma200이 baseline 대비 MDD 줄인 크래시 {wins}/{tot}")
    logger.info("=" * 80)

    p = REPORTS / "korea_long_history.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "markets": out_all}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
