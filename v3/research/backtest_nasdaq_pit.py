"""NASDAQ reversion 엔진 — survivorship-free 재검증 (V4, 2026-05-29).

이전 `test_nasdaq_engine.py` 는 yfinance survivor-only (현재까지 살아남은 76종목).
→ reversion 부풀림: 과매도된 것 중 상폐(파산)된 종목 누락, 반등한 것만 보임.

이 스크립트는 EODHD 상폐 포함 universe 로 부풀림 제거:
  - universe pool: NASDAQ Common Stock active(4097) ∪ delisted(10459) — survivorship-free
  - 데이터: EODHD /eod adjusted_close(수익률용) + raw close×volume(거래대금용)
  - point-in-time universe: 각 rebalance 시점 거래대금(trailing 20d) 상위 중
    mega-cap(top SKIP) 제외 → 중소형 POOL (look-ahead 없음, 한국 PIT 방식 이식)
  - 전략: reversion. past(lb)일 수익률 하위 QUANTILE(과매도) → long, HOLD일 보유
  - 방향: long-only (survivor-only 탐색에서 short 기각 — 성장주 과매수 momentum 잔존)
  - 상폐 손실 반영(핵심): hold 중 데이터 종료 시 last-valid 가격으로 실현,
    종료(상폐)면 추가 패널티. 떨어지는 칼을 long 하면 반등 못 하고 상폐되는 케이스 포착.
  - VIX filter: 고변동성(과민반응) regime 에서만 진입 — survivor-only 에서 MDD 절반

비교 기준 (survivor-only, reports/nasdaq_engine.json):
  lb10 long_only  annual +30.0% Sharpe 0.64 MDD -41.6%
  lb10 long+VIX   annual +14.7% Sharpe 0.67 MDD -22.9%
목표: 부풀림 제거 후에도 long_only / VIX filter 가 살아남는지 + 실제 수치.

Usage:
    PYTHONPATH=. python v3/research/backtest_nasdaq_pit.py --build   # 캐시만 (background)
    PYTHONPATH=. python v3/research/backtest_nasdaq_pit.py           # 백테스트
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")
load_dotenv()

BASE = "https://eodhd.com/api"
TOKEN = os.getenv("EODHD_API_KEY")

START = "2021-06-01"
END = "2026-05-01"
MIN_DAYS = 150          # window 내 최소 거래일 (미달 = pre-window 상폐 → 제외)

# 전략 파라미터
HOLD = 5
QUANTILE = 0.2          # 하위 20% 과매도 → long
SKIP_TOP = 30           # 거래대금 상위 N (mega/large-cap) 제외 → 중소형 타겟
POOL = 300              # 그 다음 거래대금 상위 POOL 가 point-in-time universe
COST = 0.001            # 미국 편도
DELIST_PEN = 0.30       # 상폐(hold 중 종료) 추가 손실 (last-valid 위에 보수적 가산)
MIN_PRICE = 3.0         # 진입가 floor (penny/상폐직전 0원/글리치 제외, 중소형 의도)
RET_CAP = 5.0           # 단일 종목 수익률 상한 (bad-print 글리치 가드)
REBAL_PER_YEAR = 252 / HOLD

REPORTS = Path("v3/research/reports")
SYMS = REPORTS / "nasdaq_symbols.json"
CACHE = REPORTS / "nasdaq_pit_cache.parquet"


class Rate:
    """thread-safe 토큰버킷 (EODHD 1000/min 한도 아래 유지)."""
    def __init__(self, per_min: int = 850):
        self.interval = 60.0 / per_min
        self.lock = threading.Lock()
        self.nxt = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            sleep = max(0.0, self.nxt - now)
            self.nxt = max(now, self.nxt) + self.interval
        if sleep > 0:
            time.sleep(sleep)


def nasdaq_common_codes() -> list[str]:
    if SYMS.exists():
        return json.loads(SYMS.read_text())
    logger.info("Fetching NASDAQ Common Stock symbol lists (active + delisted)...")
    act = requests.get(f"{BASE}/exchange-symbol-list/US",
                       params={"api_token": TOKEN, "fmt": "json"}, timeout=180).json()
    dl = requests.get(f"{BASE}/exchange-symbol-list/US",
                      params={"api_token": TOKEN, "delisted": 1, "fmt": "json"}, timeout=180).json()
    def keep(rows):
        return [r["Code"] for r in rows
                if r.get("Type") == "Common Stock" and r.get("Exchange") == "NASDAQ"]
    codes = sorted(set(keep(act)) | set(keep(dl)))
    SYMS.write_text(json.dumps(codes))
    logger.info(f"  {len(codes)} NASDAQ common stocks (survivorship-free pool)")
    return codes


def fetch_one(code: str, rate: Rate) -> pd.DataFrame | None:
    for attempt in range(3):
        rate.wait()
        try:
            r = requests.get(f"{BASE}/eod/{code}.US",
                             params={"api_token": TOKEN, "fmt": "json", "from": START, "to": END},
                             timeout=60)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1)); continue
            if not r.ok:
                return None
            j = r.json()
            if not isinstance(j, list) or len(j) < MIN_DAYS:
                return None
            df = pd.DataFrame(j)
            if "adjusted_close" not in df or "volume" not in df:
                return None
            df = df[["date", "adjusted_close", "close", "volume"]].dropna()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["ticker"] = code
            df["dollarvol"] = df["close"] * df["volume"]   # 거래대금 (raw close)
            return df[["date", "ticker", "adjusted_close", "dollarvol"]].rename(
                columns={"adjusted_close": "adj"})
        except Exception:
            time.sleep(1.0)
    return None


def build_cache(workers: int = 12) -> pd.DataFrame:
    if CACHE.exists():
        logger.info(f"Loading cache: {CACHE}")
        return pd.read_parquet(CACHE)
    codes = nasdaq_common_codes()
    rate = Rate(per_min=850)
    frames, done, ok = [], 0, 0
    logger.info(f"Fetching {len(codes)} tickers (workers={workers}, ~{len(codes)/850:.0f} min)...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c, rate): c for c in codes}
        for fut in as_completed(futs):
            done += 1
            df = fut.result()
            if df is not None:
                frames.append(df); ok += 1
            if done % 500 == 0:
                logger.info(f"  {done}/{len(codes)}  (ok={ok})")
    panel = pd.concat(frames, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE)
    logger.info(f"  cached {len(panel)} rows, {panel['ticker'].nunique()} tickers in window")
    return panel


def backtest(adj: pd.DataFrame, dvol: pd.DataFrame, lb: int,
             vix: pd.Series | None = None, vix_filter: bool = False,
             vix_pctl: float = 0.5, delist_pen: float = DELIST_PEN) -> dict:
    dates = adj.index.tolist()
    vix_thresh = vix.quantile(vix_pctl) if vix is not None else None
    # trailing 20d 평균 거래대금 (point-in-time 유동성)
    liq20 = dvol.rolling(20, min_periods=10).mean()
    rets, prev = [], set()
    for i in range(lb, len(dates) - HOLD, HOLD):
        d = dates[i]
        if vix_filter and vix is not None:
            vv = vix.asof(d)
            if pd.isna(vv) or vv < vix_thresh:
                rets.append(0.0); prev = set(); continue
        liq = liq20.iloc[i].dropna()
        if len(liq) < SKIP_TOP + 30:
            rets.append(0.0); prev = set(); continue
        # mega-cap 제외 → 중소형 POOL
        ranked = liq.sort_values(ascending=False)
        pool = ranked.iloc[SKIP_TOP:SKIP_TOP + POOL].index
        pi, pl = adj.iloc[i][pool], adj.iloc[i - lb][pool]
        past = (pi / pl - 1.0)
        # finite + 진입가 floor (penny/0원/글리치 제외)
        valid = np.isfinite(past) & (pi >= MIN_PRICE) & (pl > 0)
        past = past[valid]
        if len(past) < 20:
            rets.append(0.0); prev = set(); continue
        k = max(int(len(past) * QUANTILE), 3)
        losers = past.nsmallest(k).index            # 과매도 → long
        pr = []
        for t in losers:
            e = adj.iloc[i][t]
            x = adj.iloc[i + HOLD][t]
            if pd.notna(x):
                pr.append(min(x / e - 1.0, RET_CAP))
            else:                                    # hold 중 상폐
                w = adj.iloc[i:i + HOLD + 1][t]
                lv = w.last_valid_index()
                if lv is not None and w[lv] > 0:
                    pr.append(w[lv] / e - 1.0 - delist_pen)
                else:
                    pr.append(-1.0)
        gross = float(np.mean(pr))
        turnover = len(set(losers) - prev) / max(len(losers), 1)
        rets.append(gross - COST * turnover)
        prev = set(losers)
    rets = np.array(rets)
    if len(rets) == 0:
        return {}
    if not np.all(np.isfinite(rets)):
        return {"annual": float("nan"), "sharpe": 0.0, "mdd": float("nan"),
                "win": float("nan"), "n": len(rets), "error": "non-finite rets"}
    eq = np.cumprod(1 + rets)
    total = float(eq[-1] - 1)
    ann = float((1 + total) ** (REBAL_PER_YEAR / len(rets)) - 1) if total > -1 else -1.0
    vol = float(rets.std() * np.sqrt(REBAL_PER_YEAR))
    sharpe = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sharpe, "mdd": mdd,
            "win": float((rets > 0).mean()), "n": len(rets)}


def fetch_vix() -> pd.Series | None:
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=START, end=END, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        s = df["Close"].copy(); s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="캐시만 구축하고 종료")
    args = ap.parse_args()

    panel = build_cache()
    if args.build:
        logger.info("build-only 완료.")
        return 0

    adj = panel.pivot_table(index="date", columns="ticker", values="adj").sort_index()
    dvol = panel.pivot_table(index="date", columns="ticker", values="dollarvol").sort_index()
    vix = fetch_vix()
    logger.info(f"pool: {adj.shape[1]} tickers, {adj.shape[0]} days, VIX={'ok' if vix is not None else 'fail'}")

    logger.info("=" * 72)
    logger.info(f"NASDAQ reversion — survivorship-free (PIT 거래대금 top {SKIP_TOP}~{SKIP_TOP+POOL}, HOLD={HOLD})")
    logger.info("=" * 72)
    grid = {}
    for lb in [5, 10]:
        lo = backtest(adj, dvol, lb)
        lov = backtest(adj, dvol, lb, vix=vix, vix_filter=True, vix_pctl=0.5)
        grid[f"lb{lb}"] = {"long_only": lo, "long_only_vix": lov}
        logger.info(f"--- lookback {lb}d (reversion, long-only) ---")
        logger.info(f"  no filter : annual={lo['annual']:+.1%}  Sharpe={lo['sharpe']:+.2f}  MDD={lo['mdd']:.1%}  win={lo['win']:.0%}  n={lo['n']}")
        logger.info(f"  VIX>median: annual={lov['annual']:+.1%}  Sharpe={lov['sharpe']:+.2f}  MDD={lov['mdd']:.1%}  win={lov['win']:.0%}  n={lov['n']}")
        logger.info("")

    logger.info("VERDICT (survivorship-free):")
    best = max((v for lb in grid.values() for v in lb.values()), key=lambda x: x.get("sharpe", -9))
    logger.info(f"  best Sharpe={best['sharpe']:.2f} annual={best['annual']:+.1%} MDD={best['mdd']:.1%}")
    logger.info(f"  (survivor-only 비교: lb10 long_only +30.0%/0.64, lb10+VIX +14.7%/0.67)")
    logger.info("=" * 72)

    out = REPORTS / "nasdaq_pit_backtest.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_tickers": int(adj.shape[1]), "days": int(adj.shape[0]),
        "skip_top": SKIP_TOP, "pool": POOL, "hold": HOLD, "quantile": QUANTILE,
        "cost": COST, "delist_pen": DELIST_PEN, "grid": grid,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
