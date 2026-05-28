"""Phase 0: Crypto universe edge 검증 (V4 도메인 전환, 2026-05-29).

가설: NASDAQ-100 large-cap은 efficient (alpha IC 0.02~0.03). crypto는 비효율
(24/7, 리테일 주도, 기관 침투 얕음) → momentum/trend가 더 강한 IC.
또 perp funding rate carry는 주식에 없는 구조적 edge.

검증:
  1. Binance USDT perp top 30 daily OHLCV (3년) + funding rate
  2. directional alpha (trend/reversion/volume_surprise/breakout_fade/rsi_reversal)
     를 여러 forward horizon (1/3/5/7d)에서 cross-sectional IC 측정
  3. funding rate carry IC (funding 음수 → long 유리 가설)
  4. NASDAQ baseline (trend 0.02, breakout_fade 0.02, volume_surprise 0.03) 대비

go/no-go: crypto alpha IC가 NASDAQ의 2~3배(0.05+)면 도메인 전환 진행.
비슷하면 재고. funding carry가 robust하면 추가 edge.

데이터: Binance public API (requests, ccxt 불필요). 한국 IP 접근 확인됨.

Usage:
    PYTHONPATH=. python v3/research/test_crypto_alphas.py
"""

from __future__ import annotations

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.strategy.alpha_sources import (
    AlphaBreakoutFade,
    AlphaRSIReversal,
    AlphaReversion,
    AlphaTrend,
    AlphaVolumeSurprise,
)

FAPI = "https://fapi.binance.com"

# Binance USDT-perp major 30 (by liquidity/marketcap, 2026 기준 대표)
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "ATOMUSDT", "UNIUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "NEARUSDT", "AAVEUSDT", "TIAUSDT",
    "SEIUSDT", "LDOUSDT", "RNDRUSDT", "FETUSDT", "MKRUSDT", "STXUSDT",
]

HORIZONS = [1, 3, 5, 7]
BASELINE_NASDAQ = {  # 5d vanilla IC
    "trend": -0.002, "reversion": 0.004, "volume_surprise": 0.030,
    "breakout_fade": 0.020, "rsi_reversal": 0.015,
}


def fetch_klines(symbol: str, days: int = 1095) -> pd.DataFrame:
    """Binance perp daily klines (UTC). 최대 1500 bars/req."""
    url = f"{FAPI}/fapi/v1/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": "1d", "limit": min(days, 1500)}, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "qvol", "trades", "tbav", "tqav", "ignore",
    ])
    df["date"] = pd.to_datetime(df["openTime"], unit="ms").dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["ticker"] = symbol
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def fetch_funding(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Binance perp funding rate history (8h interval → daily mean)."""
    url = f"{FAPI}/fapi/v1/fundingRate"
    r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.normalize()
    df["fundingRate"] = df["fundingRate"].astype(float)
    daily = df.groupby("date")["fundingRate"].sum().reset_index()  # 일 3회 합
    daily["ticker"] = symbol
    return daily[["date", "ticker", "fundingRate"]]


def measure_alpha_ic(panel: pd.DataFrame, horizon: int) -> dict:
    from scipy.stats import spearmanr
    alphas = [AlphaTrend(), AlphaReversion(), AlphaVolumeSurprise(),
              AlphaBreakoutFade(), AlphaRSIReversal()]
    dates = sorted(panel["date"].unique())
    rebal = dates[60:-horizon:horizon]
    close_piv = panel.pivot_table(index="date", columns="ticker", values="close")
    ic_acc = {a.name: [] for a in alphas}
    for d in rebal:
        upto = panel[panel["date"] <= d]
        fi = dates.index(d) + horizon
        if fi >= len(dates):
            continue
        ret = (close_piv.loc[dates[fi]] / close_piv.loc[d] - 1.0).dropna()
        if len(ret) < 8:
            continue
        for a in alphas:
            sig = a.compute(upto)
            common = sig.index.intersection(ret.index)
            if len(common) < 8:
                continue
            s, r = sig.loc[common].to_numpy(), ret.loc[common].to_numpy()
            if np.std(s) < 1e-12 or np.std(r) < 1e-12:
                continue
            rho, _ = spearmanr(s, r)
            if np.isfinite(rho):
                ic_acc[a.name].append(float(rho))
    return {n: (float(np.mean(v)) if v else 0.0) for n, v in ic_acc.items()}


def measure_funding_carry(panel: pd.DataFrame, funding: pd.DataFrame, horizon: int = 1) -> dict:
    """Funding carry: 음수 funding(=long에 보조금) → forward long return 예측?
    cross-sectional IC of (−funding) vs forward return."""
    from scipy.stats import spearmanr
    merged = panel.merge(funding, on=["date", "ticker"], how="inner")
    dates = sorted(merged["date"].unique())
    close_piv = merged.pivot_table(index="date", columns="ticker", values="close")
    fund_piv = merged.pivot_table(index="date", columns="ticker", values="fundingRate")
    ics = []
    for i, d in enumerate(dates[:-horizon]):
        if d not in fund_piv.index:
            continue
        fund = fund_piv.loc[d].dropna()
        ret = (close_piv.loc[dates[i + horizon]] / close_piv.loc[d] - 1.0).dropna()
        common = fund.index.intersection(ret.index)
        if len(common) < 8:
            continue
        # 가설: 음수 funding → long 유리 → signal = -funding
        s = (-fund.loc[common]).to_numpy()
        r = ret.loc[common].to_numpy()
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue
        rho, _ = spearmanr(s, r)
        if np.isfinite(rho):
            ics.append(float(rho))
    return {"funding_carry_ic": float(np.mean(ics)) if ics else 0.0, "n_dates": len(ics)}


def main() -> int:
    logger.info(f"Fetching {len(UNIVERSE)} Binance perp OHLCV + funding (3y)...")
    ohlcv_frames, fund_frames, kept = [], [], []
    for i, sym in enumerate(UNIVERSE, 1):
        kl = fetch_klines(sym)
        if kl.empty or len(kl) < 300:
            logger.warning(f"[{i}/{len(UNIVERSE)}] {sym}: insufficient klines")
            continue
        fn = fetch_funding(sym)
        ohlcv_frames.append(kl)
        if not fn.empty:
            fund_frames.append(fn)
        kept.append(sym)
        logger.info(f"[{i}/{len(UNIVERSE)}] {sym}: {len(kl)} bars, funding {len(fn)}")
        time.sleep(0.2)

    if not ohlcv_frames:
        logger.error("No data")
        return 1
    panel = pd.concat(ohlcv_frames, ignore_index=True)
    funding = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()
    logger.info(f"Universe: {len(kept)} coins, {len(panel)} OHLCV rows")

    logger.info("=" * 64)
    logger.info("CRYPTO ALPHA IC by horizon (vs NASDAQ baseline)")
    logger.info("=" * 64)
    all_ic = {}
    for h in HORIZONS:
        ic = measure_alpha_ic(panel, h)
        all_ic[h] = ic
        logger.info(f"--- horizon {h}d ---")
        for name, v in ic.items():
            base = BASELINE_NASDAQ.get(name, 0)
            mark = " ⭐" if abs(v) >= 0.05 else (" ✓" if abs(v) >= 0.03 else "")
            logger.info(f"  {name:18s} crypto={v:+.4f}  nasdaq_5d={base:+.4f}{mark}")
        logger.info("")

    carry = measure_funding_carry(panel, funding) if not funding.empty else {"funding_carry_ic": None}
    logger.info(f"Funding carry IC (−funding vs fwd 1d): {carry}")

    logger.info("=" * 64)
    logger.info("VERDICT:")
    best = max((abs(v) for ic in all_ic.values() for v in ic.values()), default=0)
    if best >= 0.05:
        v = f"PASS — best |IC|={best:.4f} ≥ 0.05 (NASDAQ 2x+). 도메인 전환 진행 검토"
    elif best >= 0.035:
        v = f"MARGINAL — best |IC|={best:.4f} (NASDAQ보다 약간 높음). 추가 검증"
    else:
        v = f"WEAK — best |IC|={best:.4f} < 0.035 (NASDAQ과 비슷). crypto도 efficient?"
    logger.info(f"  {v}")
    logger.info("=" * 64)

    out = Path("v3/research/reports/crypto_alpha_ic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe": kept, "n_coins": len(kept), "ohlcv_rows": len(panel),
        "alpha_ic_by_horizon": {str(h): ic for h, ic in all_ic.items()},
        "funding_carry": carry,
        "baseline_nasdaq_5d": BASELINE_NASDAQ,
        "best_abs_ic": best,
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
