"""OOS validation of calibration table (PR-2.1).

Loads the panel + calibration_table.json, applies calibration to the OOS
portion (panel rows AFTER train_end), and computes:
  - Decile monotonicity (Spearman corr of decile_index vs realized mean fwd_5d)
  - Top decile vs bottom decile spread
  - Win rate and PF per decile
  - Markdown report

Pass/fail criteria (V3.3_AB_PLAN.md §6.1 PR-2.1 합격 기준):
  - decile monotonicity Spearman > 0.5
  - top decile mean fwd_5d > 0 (cost 회수 가능성)
  - top decile > bottom decile (no overlap)

CLI:
  $PYTHON v3/research/validate_edge.py \\
      --panel data/research/edge_dataset_2026-06.parquet \\
      --calibration v3/config/edge_calibration_2026-06.json \\
      --train-end 2025-04-30 \\
      --output research/reports/edge_validation_2026-06.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.research.calibrate_edge import assign_deciles_to_panel
from v3.strategy.diagnostics import _spearman
from v3.strategy.edge_calibrator import (
    DECILE_COUNT,
    load_calibration_table,
)


# ──────────────────────────────────────────────────────────────
# Validation result
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DecileStats:
    decile: int
    n: int
    mean_fwd_5d: float
    median_fwd_5d: float
    win_rate_5d: float
    profit_factor: float
    mean_calibrated: float


@dataclass(frozen=True)
class ValidationResult:
    decile_stats: list[DecileStats]
    decile_monotonicity_spearman: float
    top_minus_bottom_decile_mean: float
    top_decile_mean_positive: bool
    overall_n: int
    train_end: Optional[str]

    @property
    def passed(self) -> bool:
        """Pass criteria from V3.3_AB_PLAN.md §6.1."""
        return (
            self.decile_monotonicity_spearman > 0.5
            and self.top_decile_mean_positive
            and self.top_minus_bottom_decile_mean > 0.0
        )


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────
def _profit_factor(returns: pd.Series) -> float:
    pos = returns[returns > 0].sum()
    neg = -returns[returns < 0].sum()
    if neg <= 1e-9:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def compute_decile_stats(panel: pd.DataFrame) -> list[DecileStats]:
    """Compute per-decile statistics. panel must have _decile column."""
    stats: list[DecileStats] = []
    for d in range(DECILE_COUNT):
        group = panel[panel["_decile"] == d]
        if group.empty:
            continue
        fwd = group["forward_return_5d"]
        mean_calib = (
            float(group["expected_return_5d"].mean())
            if "expected_return_5d" in group.columns
            else 0.0
        )
        stats.append(DecileStats(
            decile=d,
            n=len(group),
            mean_fwd_5d=float(fwd.mean()),
            median_fwd_5d=float(fwd.median()),
            win_rate_5d=float((fwd > 0).mean()),
            profit_factor=_profit_factor(fwd),
            mean_calibrated=mean_calib,
        ))
    return stats


def validate_oos(
    panel_path: Path,
    calibration_path: Path,
    train_end: Optional[str] = None,
) -> ValidationResult:
    """Apply calibration to OOS panel and compute pass/fail metrics.

    Args:
        panel_path: full panel parquet
        calibration_path: trained calibration_table.json
        train_end: panel rows AFTER this date are OOS. If None, validates
            on entire panel (in-sample, weaker signal).
    """
    panel = pd.read_parquet(panel_path)
    table, metadata = load_calibration_table(calibration_path)
    decile_bp = list(metadata["decile_breakpoints"])

    if train_end:
        cutoff = pd.Timestamp(train_end)
        oos = panel[panel["date"] > cutoff].copy()
    else:
        oos = panel.copy()

    logger.info(f"OOS panel: {len(oos)} rows (train_end={train_end})")

    if oos.empty:
        logger.warning("OOS panel empty — validation cannot proceed")
        return ValidationResult(
            decile_stats=[],
            decile_monotonicity_spearman=0.0,
            top_minus_bottom_decile_mean=0.0,
            top_decile_mean_positive=False,
            overall_n=0,
            train_end=train_end,
        )

    oos["_decile"] = assign_deciles_to_panel(oos, decile_bp)

    # Build expected_return_5d column from calibration table (Level 3 only —
    # decile-level lookup, simplest baseline for monotonicity check)
    expected = []
    for _, row in oos.iterrows():
        key = (int(row["_decile"]),)
        bucket = table.get(key) or table.get(("global",))
        expected.append(bucket.mean_forward_return_5d if bucket else 0.0)
    oos["expected_return_5d"] = expected

    stats = compute_decile_stats(oos)
    if not stats:
        return ValidationResult(
            decile_stats=[], decile_monotonicity_spearman=0.0,
            top_minus_bottom_decile_mean=0.0,
            top_decile_mean_positive=False,
            overall_n=0, train_end=train_end,
        )

    # Spearman: decile_index vs mean realized forward return
    decile_idx = [s.decile for s in stats]
    mean_returns = [s.mean_fwd_5d for s in stats]
    monotonicity = _spearman(decile_idx, mean_returns)

    top_mean = stats[-1].mean_fwd_5d
    bot_mean = stats[0].mean_fwd_5d

    return ValidationResult(
        decile_stats=stats,
        decile_monotonicity_spearman=monotonicity,
        top_minus_bottom_decile_mean=top_mean - bot_mean,
        top_decile_mean_positive=top_mean > 0,
        overall_n=len(oos),
        train_end=train_end,
    )


# ──────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────
def render_markdown_report(result: ValidationResult) -> str:
    """Render validation result as markdown."""
    lines: list[str] = []
    lines.append("# Edge Calibration OOS Validation Report")
    lines.append("")
    lines.append(f"- train_end: {result.train_end}")
    lines.append(f"- OOS sample: {result.overall_n} rows")
    lines.append(
        f"- decile monotonicity (Spearman): "
        f"{result.decile_monotonicity_spearman:.3f}"
    )
    lines.append(
        f"- top - bottom decile mean fwd_5d: "
        f"{result.top_minus_bottom_decile_mean:.4f}"
    )
    lines.append(f"- top decile mean > 0: {result.top_decile_mean_positive}")
    lines.append("")
    status = "PASS" if result.passed else "FAIL"
    lines.append(f"## Pass/Fail: **{status}**")
    lines.append("")
    lines.append("Criteria (V3.3_AB_PLAN.md §6.1):")
    lines.append("- monotonicity Spearman > 0.5")
    lines.append("- top decile mean > 0")
    lines.append("- top - bottom > 0")
    lines.append("")
    lines.append("## Decile Statistics")
    lines.append("")
    lines.append("| Decile | N | Mean fwd_5d | Median | Win rate | PF | Calibrated mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for s in result.decile_stats:
        lines.append(
            f"| {s.decile} | {s.n} | {s.mean_fwd_5d:+.4f} | "
            f"{s.median_fwd_5d:+.4f} | {s.win_rate_5d:.2%} | "
            f"{s.profit_factor:.2f} | {s.mean_calibrated:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = validate_oos(
        panel_path=args.panel,
        calibration_path=args.calibration,
        train_end=args.train_end,
    )
    md = render_markdown_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    logger.info(f"Validation report saved: {args.output}")
    logger.info(
        f"PASSED" if result.passed else f"FAILED: "
        f"monotonicity={result.decile_monotonicity_spearman:.3f}, "
        f"top_positive={result.top_decile_mean_positive}, "
        f"spread={result.top_minus_bottom_decile_mean:.4f}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
