"""Build calibration table from edge panel (PR-2.1).

Steps:
  1. Load panel parquet (from build_edge_dataset.py)
  2. Compute decile breakpoints from raw_opportunity distribution
  3. Aggregate forward returns into buckets at 3 fallback levels:
       (decile, regime, liquidity_bucket)
       (decile, regime)
       (decile,)
       ("global",)
     (Level 4 sector excluded — sample 부족)
  4. Save as calibration_table.json + decile_breakpoints metadata

CLI:
  $PYTHON v3/research/calibrate_edge.py \\
      --panel data/research/edge_dataset_2026-06.parquet \\
      --train-end 2025-04-30 \\
      --output v3/config/edge_calibration_2026-06.json

tier_thresholds.json is a separate artifact derived in PR-2.3 (EdgeTier),
not in this script — depends on EdgeEngine net_edge calculation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.strategy.edge_calibrator import (
    DECILE_COUNT,
    DEFAULT_MIN_BUCKET_N,
    save_calibration_table,
)
from v3.strategy.types import CalibrationBucket


def compute_decile_breakpoints(
    panel: pd.DataFrame,
    column: str = "raw_opportunity",
) -> list[float]:
    """Compute 9 decile cut points (10th, 20th, ..., 90th percentile)."""
    if panel.empty or column not in panel.columns:
        raise ValueError(f"Panel empty or missing column {column}")
    quantiles = [(i + 1) / DECILE_COUNT for i in range(DECILE_COUNT - 1)]
    breakpoints = panel[column].quantile(quantiles).to_list()
    return [float(b) for b in breakpoints]


def assign_deciles_to_panel(
    panel: pd.DataFrame,
    decile_breakpoints: list[float],
    column: str = "raw_opportunity",
) -> pd.Series:
    """Vectorized decile assignment for the whole panel."""
    bp = np.asarray(decile_breakpoints, dtype=float)
    return pd.Series(
        np.digitize(panel[column].to_numpy(), bp, right=False),
        index=panel.index,
    )


def aggregate_bucket(group: pd.DataFrame) -> Optional[CalibrationBucket]:
    """Compute CalibrationBucket from a panel slice. None if insufficient samples (>0)."""
    n = len(group)
    if n == 0:
        return None
    fwd = group["forward_return_5d"]
    return CalibrationBucket(
        key=tuple(),  # caller fills key
        n=int(n),
        mean_forward_return_5d=float(fwd.mean()),
        mean_mae_5d=float(group["mae_5d"].mean()),
        mean_mfe_5d=float(group["mfe_5d"].mean()),
        win_rate_5d=float((fwd > 0).mean()),
        std_forward_return_5d=float(fwd.std(ddof=0)),
        median_forward_return_5d=float(fwd.median()),
    )


def _bucket_with_key(bucket: CalibrationBucket, key: tuple) -> CalibrationBucket:
    """Replace key field on bucket (frozen → new instance)."""
    return CalibrationBucket(
        key=key,
        n=bucket.n,
        mean_forward_return_5d=bucket.mean_forward_return_5d,
        mean_mae_5d=bucket.mean_mae_5d,
        mean_mfe_5d=bucket.mean_mfe_5d,
        win_rate_5d=bucket.win_rate_5d,
        std_forward_return_5d=bucket.std_forward_return_5d,
        median_forward_return_5d=bucket.median_forward_return_5d,
    )


def build_calibration_table(
    panel: pd.DataFrame,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
) -> tuple[dict[tuple, CalibrationBucket], list[float]]:
    """Build full bucket table at all fallback levels.

    Args:
        panel: train panel (post-build_edge_dataset, train period only)
        min_bucket_n: skip buckets with n < this threshold

    Returns:
        (table, decile_breakpoints)
    """
    if panel.empty:
        raise ValueError("Cannot calibrate empty panel")

    required = ["raw_opportunity", "regime", "liquidity_bucket",
                "forward_return_5d", "mae_5d", "mfe_5d"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    decile_bp = compute_decile_breakpoints(panel)
    panel = panel.copy()
    panel["_decile"] = assign_deciles_to_panel(panel, decile_bp)

    table: dict[tuple, CalibrationBucket] = {}

    # Global bucket
    global_b = aggregate_bucket(panel)
    if global_b and global_b.n >= min_bucket_n:
        table[("global",)] = _bucket_with_key(global_b, ("global",))

    # Level 3: (decile,)
    for dec, group in panel.groupby("_decile"):
        b = aggregate_bucket(group)
        if b and b.n >= min_bucket_n:
            table[(int(dec),)] = _bucket_with_key(b, (int(dec),))

    # Level 2: (decile, regime)
    for (dec, reg), group in panel.groupby(["_decile", "regime"]):
        b = aggregate_bucket(group)
        if b and b.n >= min_bucket_n:
            key = (int(dec), str(reg))
            table[key] = _bucket_with_key(b, key)

    # Level 1: (decile, regime, liquidity_bucket)
    for (dec, reg, liq), group in panel.groupby(
        ["_decile", "regime", "liquidity_bucket"]
    ):
        b = aggregate_bucket(group)
        if b and b.n >= min_bucket_n:
            key = (int(dec), str(reg), str(liq))
            table[key] = _bucket_with_key(b, key)

    logger.info(
        f"build_calibration_table: {len(table)} buckets "
        f"(global=1, levels: 3+2+1)"
    )
    return table, decile_bp


def calibrate_panel(
    panel_path: Path,
    output_path: Path,
    train_end: Optional[str] = None,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
) -> dict:
    """End-to-end: load panel → calibrate → save JSON.

    Returns:
        summary dict
    """
    panel = pd.read_parquet(panel_path)
    logger.info(f"Loaded panel: {len(panel)} rows from {panel_path}")

    if train_end:
        cutoff = pd.Timestamp(train_end)
        train_panel = panel[panel["date"] <= cutoff]
        logger.info(
            f"Train split: {len(train_panel)} rows (≤ {train_end}), "
            f"OOS: {len(panel) - len(train_panel)} rows"
        )
    else:
        train_panel = panel

    table, decile_bp = build_calibration_table(train_panel, min_bucket_n)

    metadata = {
        "version": pd.Timestamp.now().strftime("v3.3-%Y-%m-%d"),
        "panel_path": str(panel_path),
        "train_end": train_end,
        "panel_size": int(len(panel)),
        "train_size": int(len(train_panel)),
        "min_bucket_n": min_bucket_n,
        "decile_breakpoints": decile_bp,
        "regime_counts": {
            str(k): int(v)
            for k, v in train_panel["regime"].value_counts().items()
        },
    }

    save_calibration_table(table, output_path, metadata=metadata)
    logger.info(f"Saved calibration table to {output_path}")

    return {
        "n_buckets": len(table),
        "n_panel_rows": len(panel),
        "n_train_rows": len(train_panel),
        "decile_breakpoints": decile_bp,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--train-end", type=str, default=None,
                   help="YYYY-MM-DD (panel rows after this date held out)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-bucket-n", type=int, default=DEFAULT_MIN_BUCKET_N)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = calibrate_panel(
        panel_path=args.panel,
        output_path=args.output,
        train_end=args.train_end,
        min_bucket_n=args.min_bucket_n,
    )
    logger.info(f"Calibration complete: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
