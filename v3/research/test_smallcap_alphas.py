"""Small-cap universe alpha IC 검증 (V4 철학 재검토, 2026-05-28).

가설: NASDAQ-100 large-cap은 efficient → alpha IC ceiling 0.02~0.03.
small/mid-cap은 비효율 → 같은 alpha가 더 높은 IC를 가질 것.

방법:
  1. 대표 small/mid-cap 후보 다운로드 (yfinance, 3년 OHLCV + marketCap)
  2. marketCap < cap_max 필터 (진짜 small/mid만)
  3. directional alpha (trend/reversion/volume_surprise/breakout_fade/rsi_reversal)
     를 weekly rebalance date마다 cross-sectional 계산
  4. forward 5d return과 cross-sectional Spearman IC
  5. large-cap baseline (NASDAQ-100, IC 0.02~0.03)과 비교

주의 (KRX 1% 교훈): gross IC가 높아도 small-cap 비용(0.5~1%)이 잠식 가능.
이 스크립트는 GROSS IC만 측정 — net 평가는 후속 (비용 gate backtest).

Usage:
    PYTHONPATH=. python v3/research/test_smallcap_alphas.py --cap-max 3e9
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

from v3.strategy.alpha_sources import (
    AlphaBreakoutFade,
    AlphaRSIReversal,
    AlphaReversion,
    AlphaTrend,
    AlphaVolumeSurprise,
)

# 대표 small/mid-cap 후보 (다양한 섹터). marketCap 필터로 large-cap 제거.
CANDIDATES: tuple[str, ...] = (
    # software/tech
    "APPN", "BL", "FROG", "PD", "AI", "SMAR", "BRZE", "ASAN", "DOCN", "FSLY",
    "GTLB", "PATH", "BRZE", "AMPL", "FROG",
    # biotech
    "CRSP", "BEAM", "NTLA", "ARWR", "RARE", "HALO", "FOLD", "INSM", "KRYS", "EXAS",
    # consumer
    "CROX", "YETI", "SHAK", "FIVE", "WING", "CAKE", "PLAY", "DNUT", "BROS",
    # industrial
    "SAIA", "SITE", "AIT", "KFY",
    # financial
    "UPST", "LMND", "TREE", "OPEN",
    # semis (small)
    "AMBA", "SITM", "CEVA", "RMBS", "POWI", "INDI", "NVTS",
    # energy
    "SM", "MGY", "CHRD", "RRC", "AR", "CRK",
    # materials/healthcare
    "MP", "CENX", "ATI", "DOCS", "PRVA", "OMCL", "PGNY",
)


def download_universe(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    cap_max: float,
    cap_min: float,
    min_daily_usd: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Download OHLCV + marketCap, filter to small/mid-cap with liquidity."""
    import yfinance as yf

    frames = []
    kept: list[str] = []
    seen = set()
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                continue
            # flatten multiindex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            mc = yf.Ticker(t).info.get("marketCap", 0) or 0
            avg_usd = float((df["Close"] * df["Volume"]).mean())
            if not (cap_min <= mc <= cap_max):
                logger.info(f"{t}: marketCap ${mc/1e9:.1f}B out of range — skip")
                continue
            if avg_usd < min_daily_usd:
                logger.info(f"{t}: liquidity ${avg_usd/1e6:.1f}M < min — skip")
                continue
            sub = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            sub.columns = ["date", "open", "high", "low", "close", "volume"]
            sub["ticker"] = t
            sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
            frames.append(sub)
            kept.append(t)
            logger.info(f"{t}: KEEP marketCap ${mc/1e9:.1f}B, $vol ${avg_usd/1e6:.0f}M")
        except Exception as exc:
            logger.warning(f"{t}: {type(exc).__name__} {exc}")

    if not frames:
        return pd.DataFrame(), []
    panel = pd.concat(frames, ignore_index=True)
    return panel, kept


def measure_ic(panel: pd.DataFrame, horizon: int = 5, step: int = 5) -> dict:
    """Cross-sectional Spearman IC per alpha, averaged over rebalance dates."""
    from scipy.stats import spearmanr

    alphas = [
        AlphaTrend(), AlphaReversion(), AlphaVolumeSurprise(),
        AlphaBreakoutFade(), AlphaRSIReversal(),
    ]
    dates = sorted(panel["date"].unique())
    # need history for alphas (≥60d) + forward horizon
    rebal = dates[60:-horizon:step]
    logger.info(f"Measuring IC over {len(rebal)} rebalance dates...")

    # close pivot for forward return
    close_piv = panel.pivot_table(index="date", columns="ticker", values="close")

    ic_acc: dict[str, list[float]] = {a.name: [] for a in alphas}
    for d in rebal:
        upto = panel[panel["date"] <= d]
        # forward return
        fwd_idx = dates.index(d) + horizon
        if fwd_idx >= len(dates):
            continue
        d_fwd = dates[fwd_idx]
        try:
            ret = (close_piv.loc[d_fwd] / close_piv.loc[d] - 1.0).dropna()
        except KeyError:
            continue
        if len(ret) < 10:
            continue
        for a in alphas:
            sig = a.compute(upto)
            common = sig.index.intersection(ret.index)
            if len(common) < 10:
                continue
            s = sig.loc[common].to_numpy()
            r = ret.loc[common].to_numpy()
            if np.std(s) < 1e-12 or np.std(r) < 1e-12:
                continue
            rho, _ = spearmanr(s, r)
            if np.isfinite(rho):
                ic_acc[a.name].append(float(rho))

    result = {}
    for name, ics in ic_acc.items():
        if ics:
            result[name] = {
                "mean_ic": float(np.mean(ics)),
                "n_dates": len(ics),
                "ic_std": float(np.std(ics)),
            }
        else:
            result[name] = {"mean_ic": 0.0, "n_dates": 0, "ic_std": 0.0}
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2023-05-01")
    p.add_argument("--end", type=str, default="2026-05-01")
    p.add_argument("--cap-max", type=float, default=3e9, help="max marketCap (small/mid)")
    p.add_argument("--cap-min", type=float, default=3e8, help="min marketCap (avoid micro)")
    p.add_argument("--min-daily-usd", type=float, default=10e6, help="min avg daily $volume")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--output", type=str, default="v3/research/reports/smallcap_ic.json")
    args = p.parse_args()

    logger.info(f"Downloading {len(set(CANDIDATES))} candidates, "
                f"filter: ${args.cap_min/1e9:.1f}B ≤ mcap ≤ ${args.cap_max/1e9:.1f}B, "
                f"liquidity ≥ ${args.min_daily_usd/1e6:.0f}M")
    panel, kept = download_universe(
        CANDIDATES, args.start, args.end, args.cap_max, args.cap_min, args.min_daily_usd,
    )
    if panel.empty:
        logger.error("No tickers passed filter")
        return 1

    logger.info(f"Universe: {len(kept)} small/mid-cap tickers, {len(panel)} rows")
    ic = measure_ic(panel, horizon=args.horizon)

    logger.info("=" * 60)
    logger.info("SMALL/MID-CAP ALPHA IC (gross, cross-sectional Spearman)")
    logger.info(f"Universe: {len(kept)} tickers | horizon {args.horizon}d")
    logger.info("=" * 60)
    logger.info("Baseline (NASDAQ-100 large-cap, 5d): trend -0.002, reversion +0.004,")
    logger.info("  volume_surprise +0.030, breakout_fade +0.020, rsi_reversal +0.015")
    logger.info("")
    logger.info("Small/mid-cap:")
    for name, m in ic.items():
        logger.info(f"  {name:18s} mean_IC={m['mean_ic']:+.4f}  (n={m['n_dates']}, std={m['ic_std']:.3f})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(kept),
        "tickers": kept,
        "cap_max": args.cap_max, "cap_min": args.cap_min,
        "horizon": args.horizon,
        "ic": ic,
        "baseline_largecap_5d": {
            "trend": -0.002, "reversion": 0.004,
            "volume_surprise": 0.030, "breakout_fade": 0.020, "rsi_reversal": 0.015,
        },
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
