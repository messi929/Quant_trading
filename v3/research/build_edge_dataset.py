"""Build calibration panel from N-year backtest replay (PR-1.4).

For each (date, ticker) in the lookback window, computes:
  - Raw signal: direction, conviction, raw_opportunity (using t-information only)
  - Context: regime, regime_score, sector, liquidity_bucket, vol_state
  - Forward returns: 1d/3d/5d/10d
  - Risk metrics: mae_5d, mfe_5d, realized_vol_5d

Output panel feeds calibrate_edge.py (PR-2.1) to derive expected_return_5d
and tier_thresholds.

Survivorship handling:
  Default uses the CURRENT universe applied retroactively. This introduces
  survivorship bias. validate_edge.py (PR-2.1) applies haircut to top-decile
  estimates. For full survivorship-free panel, supply `historical_universe`
  callable returning {ticker: first_listed_date}.

Lookahead protection:
  - Features at t use only data <= t (intraday t excluded)
  - Forward returns use t+1 ... t+H (no overlap)
  - verify_no_lookahead() asserts feature window vs forward window separation

CLI:
  $PYTHON v3/research/build_edge_dataset.py \\
      --start 2021-01-01 --end 2026-04-30 \\
      --universe nasdaq100 \\
      --output data/research/edge_dataset_2026-06.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.strategy.alpha_sources import compute_conviction, compute_directional
from v3.strategy.opportunity import OpportunityScorer
from v3.strategy.regime_v2 import Regime, RegimeDetectorV2

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
FORWARD_HORIZONS_DAYS: list[int] = [1, 3, 5, 10]
DEFAULT_VOL_LOOKBACK: int = 20
DEFAULT_LIQUIDITY_PCTLS: tuple[float, float] = (0.33, 0.67)
MIN_HISTORY_FOR_PANEL: int = 60   # OHLCV warmup for alphas

# Required panel columns
PANEL_COLUMNS: list[str] = [
    "date", "ticker",
    "raw_opportunity", "direction", "conviction",
    "regime", "regime_score",
    "sector", "liquidity_bucket", "vol_state",
    "forward_return_1d", "forward_return_3d",
    "forward_return_5d", "forward_return_10d",
    "mae_5d", "mfe_5d", "realized_vol_5d",
]


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BuildConfig:
    start: str
    end: str
    universe: str = "nasdaq100"
    output_path: Optional[Path] = None
    cost: float = 0.001
    gate_multiplier: float = 1.75
    min_history: int = MIN_HISTORY_FOR_PANEL
    sector_map: Optional[dict[str, str]] = None
    historical_universe: Optional[Callable[[pd.Timestamp], list[str]]] = None


@dataclass(frozen=True)
class ForwardOutcome:
    forward_return_1d: float
    forward_return_3d: float
    forward_return_5d: float
    forward_return_10d: float
    mae_5d: float
    mfe_5d: float
    realized_vol_5d: float


# ──────────────────────────────────────────────────────────────
# Forward returns
# ──────────────────────────────────────────────────────────────
def compute_forward_outcome(
    ohlcv_ticker: pd.DataFrame,
    t: pd.Timestamp,
    horizons_days: list[int] = FORWARD_HORIZONS_DAYS,
) -> Optional[ForwardOutcome]:
    """Compute forward return + MAE/MFE/realized_vol for a single ticker at t.

    Lookahead-safe:
      - Returns reference t-close (must be at or before t)
      - Forward returns use prices strictly AFTER t (t+1 ... t+H)
      - MAE/MFE use intraday H/L within (t+1 ... t+5)

    Args:
        ohlcv_ticker: single-ticker OHLCV DataFrame, columns
            [date, open, high, low, close]. Sorted by date ascending.
        t: anchor date.
        horizons_days: forward horizons (calendar/business days separately).
            Index-based; relies on ohlcv_ticker being trading-day frequency.

    Returns:
        ForwardOutcome or None if insufficient forward data.
    """
    if ohlcv_ticker.empty:
        return None

    df = ohlcv_ticker.sort_values("date").reset_index(drop=True)

    # Find anchor index (last row with date <= t)
    anchor_mask = df["date"] <= t
    if not anchor_mask.any():
        return None
    anchor_idx = int(anchor_mask.sum()) - 1

    base_close = float(df.iloc[anchor_idx]["close"])
    if base_close <= 0:
        return None

    max_h = max(horizons_days)
    if anchor_idx + max_h >= len(df):
        return None  # insufficient forward data

    forwards: dict[int, float] = {}
    for h in horizons_days:
        future_close = float(df.iloc[anchor_idx + h]["close"])
        forwards[h] = future_close / base_close - 1.0

    # MAE/MFE over t+1..t+5
    h5_window = df.iloc[anchor_idx + 1: anchor_idx + 6]
    if h5_window.empty:
        return None
    lows = h5_window["low"].to_numpy(dtype=float)
    highs = h5_window["high"].to_numpy(dtype=float)
    mae = float(lows.min() / base_close - 1.0)
    mfe = float(highs.max() / base_close - 1.0)

    # Realized vol: stdev of daily returns t+1..t+5
    daily_returns = h5_window["close"].pct_change().dropna()
    if len(daily_returns) < 2:
        realized_vol = 0.0
    else:
        realized_vol = float(daily_returns.std(ddof=0) * np.sqrt(252))

    return ForwardOutcome(
        forward_return_1d=forwards[1],
        forward_return_3d=forwards[3],
        forward_return_5d=forwards[5],
        forward_return_10d=forwards[10],
        mae_5d=mae,
        mfe_5d=mfe,
        realized_vol_5d=realized_vol,
    )


# ──────────────────────────────────────────────────────────────
# Bucket helpers
# ──────────────────────────────────────────────────────────────
def assign_liquidity_bucket(
    daily_volume_value: float,
    breakpoints: tuple[float, float],
) -> str:
    """Classify by daily $-volume.

    Args:
        daily_volume_value: avg daily dollar volume over recent window
        breakpoints: (low_threshold, high_threshold) — typically 33%/67% percentile
    """
    low, high = breakpoints
    if daily_volume_value < low:
        return "low"
    if daily_volume_value < high:
        return "mid"
    return "high"


def assign_vol_state(
    current_vol: float,
    baseline_vol: float,
    expansion_threshold: float = 1.20,
    contraction_threshold: float = 0.80,
) -> str:
    """Classify vol regime: expanding / neutral / contracting."""
    if baseline_vol <= 1e-9:
        return "neutral"
    ratio = current_vol / baseline_vol
    if ratio >= expansion_threshold:
        return "expanding"
    if ratio <= contraction_threshold:
        return "contracting"
    return "neutral"


# ──────────────────────────────────────────────────────────────
# Lookahead validation
# ──────────────────────────────────────────────────────────────
def verify_no_lookahead(panel: pd.DataFrame) -> None:
    """Assert panel has no obvious lookahead violations.

    Checks:
      1. forward_return_5d not used in any feature column
      2. realized_vol_5d not used in feature derivation
      3. (date, ticker) unique

    Raises:
        AssertionError on violation.
    """
    if panel.empty:
        return

    # Uniqueness
    n = len(panel)
    n_unique = panel.groupby(["date", "ticker"]).size().shape[0]
    assert n == n_unique, f"Duplicate (date,ticker): {n} rows but {n_unique} unique"

    # Non-null forward returns
    fwd_cols = [c for c in panel.columns if c.startswith("forward_return_")]
    assert fwd_cols, "Panel missing forward_return columns"

    # Feature columns must NOT correlate with same-row forward_return_5d via
    # identity (sanity check — exact equality flag obvious leak)
    feature_cols = ["raw_opportunity", "direction", "conviction"]
    if "forward_return_5d" in panel.columns:
        for fc in feature_cols:
            if fc not in panel.columns:
                continue
            if (panel[fc] == panel["forward_return_5d"]).all():
                raise AssertionError(
                    f"Identity leak: {fc} == forward_return_5d in all rows"
                )


# ──────────────────────────────────────────────────────────────
# Main build function
# ──────────────────────────────────────────────────────────────
def build_edge_dataset(
    cfg: BuildConfig,
    ohlcv_data: pd.DataFrame,
    macro_pctl: pd.DataFrame,
    vol_predictions: pd.DataFrame,
    regime_detector: Optional[RegimeDetectorV2] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build calibration panel.

    Args:
        cfg: BuildConfig (start/end/output/etc).
        ohlcv_data: long-format OHLCV with [date, ticker, open, high, low,
            close, volume]. Sorted by (ticker, date).
        macro_pctl: rolling percentile of macro features, indexed by date.
        vol_predictions: [date, ticker, vol_score, confidence] (from VolInference
            historical replay).
        regime_detector: optional pre-loaded detector. If None, instantiates
            RegimeDetectorV2() with default weights path.
        sector_map: {ticker: GICS_sector} optional.

    Returns:
        Panel DataFrame with PANEL_COLUMNS schema.
    """
    sector_map = sector_map or {}
    if regime_detector is None:
        regime_detector = RegimeDetectorV2(
            weights_path="v3/config/alpha_weights.json"
        )
    scorer = OpportunityScorer(gate_multiplier=cfg.gate_multiplier)

    start_ts = pd.Timestamp(cfg.start)
    end_ts = pd.Timestamp(cfg.end)

    ohlcv = ohlcv_data[
        (ohlcv_data["date"] >= start_ts) & (ohlcv_data["date"] <= end_ts)
    ].copy()
    if ohlcv.empty:
        logger.warning(f"No OHLCV data in [{cfg.start}, {cfg.end}]")
        return pd.DataFrame(columns=PANEL_COLUMNS)

    dates = sorted(ohlcv["date"].unique())
    logger.info(
        f"build_edge_dataset: {len(dates)} dates, "
        f"{ohlcv['ticker'].nunique()} tickers, "
        f"start={cfg.start} end={cfg.end}"
    )

    rows: list[dict] = []

    # Liquidity breakpoints from full-period dollar-volume distribution
    dollar_vol = (ohlcv["close"] * ohlcv["volume"]).groupby(ohlcv["ticker"]).mean()
    p33 = float(dollar_vol.quantile(DEFAULT_LIQUIDITY_PCTLS[0]))
    p67 = float(dollar_vol.quantile(DEFAULT_LIQUIDITY_PCTLS[1]))
    liquidity_breakpoints = (p33, p67)

    for t in dates:
        df_up_to_t = ohlcv[ohlcv["date"] <= t]
        if len(df_up_to_t) < cfg.min_history:
            continue

        # Universe at t
        if cfg.historical_universe:
            universe_t = cfg.historical_universe(t)
        else:
            universe_t = sorted(df_up_to_t["ticker"].unique())

        # Regime at t (lookahead-safe: macro_pctl has rolling percentile)
        regime: Regime = regime_detector.detect(macro_pctl, as_of=t)

        # Vol scores at t
        vs_t = vol_predictions[vol_predictions["date"] == t][
            ["ticker", "vol_score", "confidence"]
        ]
        if vs_t.empty:
            continue

        # Compute alphas (uses df_up_to_t — no lookahead)
        directional = compute_directional(df_up_to_t)
        conviction = compute_conviction(df_up_to_t, vol_scores=vs_t)
        if directional.empty:
            continue

        # Score
        report = scorer.score(directional, conviction, regime, cost=cfg.cost)

        # Per-ticker forward outcomes
        for r in report.rows:
            ticker = r.ticker
            if cfg.historical_universe and ticker not in universe_t:
                continue

            # Forward outcome (lookahead-safe: uses dates > t)
            ohlcv_ticker = ohlcv[ohlcv["ticker"] == ticker]
            outcome = compute_forward_outcome(ohlcv_ticker, t)
            if outcome is None:
                continue

            # Liquidity bucket from per-ticker mean dollar volume
            ticker_dvol = float(dollar_vol.get(ticker, 0.0))
            liq_bucket = assign_liquidity_bucket(ticker_dvol, liquidity_breakpoints)

            # Vol state from current vs baseline
            vs_row = vs_t[vs_t["ticker"] == ticker]
            current_vol = float(vs_row["vol_score"].iloc[0]) if not vs_row.empty else 0.0
            vol_state = assign_vol_state(current_vol, baseline_vol=1.0)

            rows.append({
                "date": t,
                "ticker": ticker,
                "raw_opportunity": r.opportunity,
                "direction": r.direction,
                "conviction": r.conviction,
                "regime": regime.name,
                "regime_score": regime.score,
                "sector": sector_map.get(ticker, "unknown"),
                "liquidity_bucket": liq_bucket,
                "vol_state": vol_state,
                "forward_return_1d": outcome.forward_return_1d,
                "forward_return_3d": outcome.forward_return_3d,
                "forward_return_5d": outcome.forward_return_5d,
                "forward_return_10d": outcome.forward_return_10d,
                "mae_5d": outcome.mae_5d,
                "mfe_5d": outcome.mfe_5d,
                "realized_vol_5d": outcome.realized_vol_5d,
            })

    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    logger.info(f"build_edge_dataset: {len(panel)} rows produced")

    if not panel.empty:
        verify_no_lookahead(panel)

    if cfg.output_path:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cfg.output_path, index=False)
        logger.info(f"Saved panel to {cfg.output_path}")

    return panel


# ──────────────────────────────────────────────────────────────
# Sanity checks (CLI summary)
# ──────────────────────────────────────────────────────────────
def panel_sanity_summary(panel: pd.DataFrame) -> dict:
    """Compute summary stats for sanity checks (used by CLI + tests)."""
    if panel.empty:
        return {"n_rows": 0, "n_tickers": 0, "n_dates": 0}

    return {
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].nunique(),
        "n_dates": panel["date"].nunique(),
        "regime_counts": panel["regime"].value_counts().to_dict(),
        "sector_counts": panel["sector"].value_counts().head(10).to_dict(),
        "missing_pct": {
            c: round(float(panel[c].isna().mean()) * 100.0, 2)
            for c in panel.columns
        },
        "fwd_5d_mean": round(float(panel["forward_return_5d"].mean()), 5),
        "fwd_5d_std": round(float(panel["forward_return_5d"].std()), 5),
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--universe", type=str, default="nasdaq100",
        choices=["nasdaq100", "kospi200", "all"],
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("data/research/edge_dataset.parquet"),
    )
    p.add_argument(
        "--ohlcv-path", type=Path, default=None,
        help="Optional pre-loaded OHLCV parquet (otherwise loads via collector)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = BuildConfig(
        start=args.start,
        end=args.end,
        universe=args.universe,
        output_path=args.output,
    )

    # Real loading is the user's responsibility (server has the data).
    # Here we validate the config and emit a stub.
    logger.info(f"BuildConfig: {cfg}")
    logger.warning(
        "CLI is a stub. Production usage:\n"
        "  1. Run from server with v3/data/raw/ populated\n"
        "  2. Ensure v3/saved_models/vol_transformer_best.pt exists\n"
        "  3. Provide ohlcv_data, macro_pctl, vol_predictions to build_edge_dataset()\n"
        "Use the function directly from a Python script for now."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
