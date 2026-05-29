"""한국 momentum 엔진 risk-overlay 실험 (V4, 2026-05-29).

검증된 PIT momentum(KOSDAQ +21.7%/0.68/MDD-47%, KOSPI +28.9%/0.66/MDD-36%) 위에
risk lever 를 얹어 MDD 를 운영가능 수준(목표 25%)으로 낮추고 Sharpe 1.0+ 달성하는지.

이번 실험: lever 1 (regime filter) + lever 2 (KOSPI+KOSDAQ 결합).
  - regime gate: 지수 추세로 risk-on/off. off 면 그 시점 현금(ret 0).
      none / sma200(지수>200d SMA) / mom60(지수 60d>0) / mom120
  - 결합: KOSPI sleeve 50% + KOSDAQ sleeve 50%, 각자 독립 gate. 한 시장 off 면
      해당 sleeve 현금 → 자동 노출 축소 + 분산.

기존 PIT 캐시 재사용 (korea_{kospi,kosdaq}_pit_cache.parquet). 신호/비용/상폐처리
전부 backtest_kosdaq_pit.py 와 동일 — risk overlay 효과만 격리 측정.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_risk_overlay.py
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

LOOKBACK = 60
HOLD = 20
LIQ_TOP = 100
N_POS = 20
COST = 0.004
DELIST_PEN = 0.5
START, END = "2021-01-01", "2026-05-01"
RPY = 252 / HOLD
REPORTS = Path("v3/research/reports")

MARKETS = {
    "KOSDAQ": {"cache": REPORTS / "korea_kosdaq_pit_cache.parquet", "index": "KQ11"},
    "KOSPI": {"cache": REPORTS / "korea_kospi_pit_cache.parquet", "index": "KS11"},
}


def load_market(cache: Path):
    panel = pd.read_parquet(cache)
    close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    return close, close * vol  # close, dollar-volume(거래대금)


def regime_series(index: pd.Series, kind: str) -> pd.Series:
    """daily 지수 → risk-on(True)/off(False) daily series."""
    if kind == "none":
        return pd.Series(True, index=index.index)
    if kind == "sma200":
        return index > index.rolling(200, min_periods=100).mean()
    if kind == "mom60":
        return (index / index.shift(60) - 1.0) > 0
    if kind == "mom120":
        return (index / index.shift(120) - 1.0) > 0
    raise ValueError(kind)


def sleeve(close, dvol, regime: pd.Series, n_pos=N_POS) -> pd.Series:
    """한 시장 sleeve — rebalance date → period return (현금=0). PIT momentum + gate."""
    dates = close.index.tolist()
    out = {}
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        d = dates[i]
        ro = regime.asof(d)
        if not (ro is True or ro == True):  # risk-off or NaN → 현금
            out[d] = 0.0
            continue
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[d] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - LOOKBACK][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            out[d] = 0.0; continue
        picks = trend.nlargest(min(n_pos, len(trend))).index
        pr = []
        for t in picks:
            e = close.iloc[i][t]; x = close.iloc[i + HOLD][t]
            if pd.notna(x):
                pr.append(x / e - 1.0)
            else:
                w = close.iloc[i:i + HOLD + 1][t]; lv = w.last_valid_index()
                pr.append((w[lv] / e - 1.0 - DELIST_PEN) if lv is not None and w[lv] > 0 else -1.0)
        out[d] = float(np.mean(pr)) - COST  # 단순화: 매 rebal full turnover 비용
    return pd.Series(out).sort_index()


def stats(rets: pd.Series) -> dict:
    r = rets.values
    if len(r) == 0:
        return {}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    cash = float((r == 0).mean())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "win": float((r > 0).mean()),
            "cash_pct": cash, "n": len(r)}


def main() -> int:
    data, idx = {}, {}
    for m, cfg in MARKETS.items():
        close, dvol = load_market(cfg["cache"])
        data[m] = (close, dvol)
        s = fdr.DataReader(cfg["index"], START, END)["Close"]
        s.index = pd.to_datetime(s.index).normalize()
        idx[m] = s
        logger.info(f"{m}: {close.shape[1]} tickers, {close.shape[0]} days, index {cfg['index']} ok")

    gates = ["none", "sma200", "mom60", "mom120"]
    logger.info("=" * 78)
    logger.info(f"한국 risk-overlay — regime gate (N={N_POS}, cost {COST:.1%}, 상폐pen {DELIST_PEN:.0%})")
    logger.info("=" * 78)

    sleeves = {}  # (market, gate) -> Series
    results = {}
    for m in MARKETS:
        close, dvol = data[m]
        logger.info(f"--- {m} (PIT baseline MDD: KOSDAQ-47%/KOSPI-36%) ---")
        for g in gates:
            reg = regime_series(idx[m], g)
            sl = sleeve(close, dvol, reg)
            sleeves[(m, g)] = sl
            st = stats(sl)
            results[f"{m}_{g}"] = st
            logger.info(f"  gate={g:7s}: annual={st['annual']:+.1%}  Sharpe={st['sharpe']:+.2f}  "
                        f"MDD={st['mdd']:.1%}  win={st['win']:.0%}  cash={st['cash_pct']:.0%}")
        logger.info("")

    # 결합: KOSPI 50% + KOSDAQ 50%, gate별
    logger.info("--- 결합 (KOSPI 50% + KOSDAQ 50%, sleeve별 독립 gate) ---")
    for g in gates:
        a = sleeves[("KOSPI", g)]; b = sleeves[("KOSDAQ", g)]
        df = pd.concat([a, b], axis=1).fillna(0.0)
        comb = 0.5 * df.iloc[:, 0] + 0.5 * df.iloc[:, 1]
        st = stats(comb)
        results[f"COMBINED_{g}"] = st
        logger.info(f"  gate={g:7s}: annual={st['annual']:+.1%}  Sharpe={st['sharpe']:+.2f}  "
                    f"MDD={st['mdd']:.1%}  win={st['win']:.0%}  cash={st['cash_pct']:.0%}")
    logger.info("")

    best = max(results.items(), key=lambda kv: kv[1].get("sharpe", -9))
    logger.info("VERDICT:")
    logger.info(f"  best: {best[0]}  Sharpe={best[1]['sharpe']:.2f} annual={best[1]['annual']:+.1%} MDD={best[1]['mdd']:.1%}")
    target = [k for k, v in results.items() if abs(v.get("mdd", -1)) <= 0.25 and v.get("sharpe", 0) >= 1.0]
    logger.info(f"  목표 달성(MDD≤25% & Sharpe≥1.0): {target if target else '없음 — 추가 lever(vol-target/trailing stop) 필요'}")
    logger.info("=" * 78)

    out = REPORTS / "korea_risk_overlay.json"
    out.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                               "n_pos": N_POS, "cost": COST, "delist_pen": DELIST_PEN,
                               "results": results}, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
