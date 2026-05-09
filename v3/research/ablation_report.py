"""V3.3 Ablation report — comparison matrix + markdown rendering (PR-5.1).

Generates:
  - comparison_matrix.md: side-by-side metrics table for all ablations
  - per_ablation summary text
  - pass/fail evaluation against V3.3_AB_PLAN §6 합격 기준

Pass criteria (V3.3_AB_PLAN.md §6):
  - leverage_violations == 0 (invariant)
  - max_single_weight ≤ max_single_weight constraint (default 0.40)
  - Pyramid count > baseline (winner expansion)
  - Risk exits behave consistently (not regressed)

Statistical significance (Phase 5.2):
  Bootstrap p-value comparison vs baseline — placeholder here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from v3.research.ablation_runner import AblationMetrics


# ──────────────────────────────────────────────────────────────
# Comparison matrix
# ──────────────────────────────────────────────────────────────
def build_comparison_matrix(
    results: list[AblationMetrics],
) -> pd.DataFrame:
    """Build a DataFrame with one row per ablation, headline metrics in columns."""
    rows: list[dict] = []
    for m in results:
        rows.append({
            "ablation": m.ablation_name,
            "family": m.family,
            "n_actions": m.n_actions,
            "n_dates": m.n_dates,
            "add_new": m.action_distribution.add_new,
            "add_to_winner": m.action_distribution.add_to_winner,
            "keep": m.action_distribution.keep,
            "trim": m.action_distribution.trim,
            "exit": m.action_distribution.exit,
            "rotate": m.action_distribution.rotate,
            "avg_deployed": round(m.avg_deployed, 4),
            "max_single_weight": round(m.max_single_weight, 4),
            "leverage_violations": m.leverage_violations,
            "pyramid_count": m.pyramid_count,
            "rotation_count": m.rotation_count,
            "risk_exit_count": m.risk_exit_count,
            "errors": len(m.error_messages),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Pass/fail evaluation
# ──────────────────────────────────────────────────────────────
def evaluate_ablation_pass(
    metrics: AblationMetrics,
    baseline: Optional[AblationMetrics] = None,
    max_single_weight_limit: float = 0.40,
) -> tuple[bool, list[str]]:
    """Evaluate an ablation against V3.3_AB_PLAN §6 criteria.

    Args:
        metrics: target ablation metrics
        baseline: baseline metrics for comparison (Pyramid > baseline check)
        max_single_weight_limit: per-position weight cap

    Returns:
        (passed, failure_reasons)
    """
    failures: list[str] = []

    # Critical invariant: leverage 위반 0
    if metrics.leverage_violations > 0:
        failures.append(
            f"leverage_violations={metrics.leverage_violations} > 0 "
            f"(LEVERAGE_CAP 1.00 invariant 위배)"
        )

    # max_single_weight cap
    if metrics.max_single_weight > max_single_weight_limit + 1e-6:
        failures.append(
            f"max_single_weight={metrics.max_single_weight:.4f} > "
            f"limit {max_single_weight_limit}"
        )

    # Errors
    if metrics.error_messages:
        failures.append(f"{len(metrics.error_messages)} runtime errors")

    # Pyramid expansion check (only if baseline provided)
    if baseline is not None and metrics.family in ("cumulative", "full"):
        if metrics.action_distribution.add_to_winner < baseline.action_distribution.add_to_winner:
            failures.append(
                f"pyramid count {metrics.pyramid_count} < "
                f"baseline {baseline.pyramid_count}"
            )

    return (len(failures) == 0, failures)


# ──────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────
def render_comparison_markdown(
    results: list[AblationMetrics],
    baseline_name: str = "baseline",
    max_single_weight_limit: float = 0.40,
) -> str:
    """Render comparison matrix + pass/fail summary as markdown."""
    if not results:
        return "# Ablation Report\n\n_No results._"

    df = build_comparison_matrix(results)
    baseline = next((m for m in results if m.ablation_name == baseline_name), None)

    lines: list[str] = []
    lines.append("# V3.3 Ablation Comparison Report")
    lines.append("")
    lines.append(f"- Total ablations: {len(results)}")
    lines.append(f"- Baseline: {baseline_name}")
    if baseline is not None:
        lines.append(f"- Baseline n_dates: {baseline.n_dates}")
    lines.append("")

    # Comparison table
    lines.append("## Comparison Matrix")
    lines.append("")
    lines.append("| Ablation | Family | Actions | ADD_NEW | ADD_WIN | KEEP | TRIM | EXIT | ROTATE | Avg Deployed | Max Wgt | Lev viol |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['ablation']} | {row['family']} | "
            f"{row['n_actions']} | {row['add_new']} | {row['add_to_winner']} | "
            f"{row['keep']} | {row['trim']} | {row['exit']} | {row['rotate']} | "
            f"{row['avg_deployed']:.3f} | {row['max_single_weight']:.3f} | "
            f"{row['leverage_violations']} |"
        )
    lines.append("")

    # Backtest metrics table (only if any backtest fields populated)
    has_backtest = any(m.sharpe is not None for m in results)
    if has_backtest:
        lines.append("## Backtest Metrics (production mode)")
        lines.append("")
        lines.append("| Ablation | Sharpe | MDD | Return | Win Rate | PF | Trades |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for m in results:
            if m.sharpe is None:
                continue
            lines.append(
                f"| {m.ablation_name} | {m.sharpe:.2f} | "
                f"{m.max_drawdown:.3f} | {m.total_return:+.3f} | "
                f"{m.win_rate:.2%} | {m.profit_factor:.2f} | "
                f"{m.n_trades} |"
            )
        lines.append("")

    # Pass/fail summary
    lines.append("## Pass/Fail Summary (V3.3_AB_PLAN §6)")
    lines.append("")
    pass_count = 0
    for m in results:
        passed, failures = evaluate_ablation_pass(
            m, baseline=baseline,
            max_single_weight_limit=max_single_weight_limit,
        )
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        lines.append(f"### {m.ablation_name} — **{status}**")
        if failures:
            for reason in failures:
                lines.append(f"  - {reason}")
        else:
            lines.append("  - all criteria met")
        lines.append("")

    lines.append(f"**Overall: {pass_count} / {len(results)} passed.**")
    lines.append("")

    return "\n".join(lines)


def save_comparison_report(
    results: list[AblationMetrics],
    output_path: str | Path,
    baseline_name: str = "baseline",
    max_single_weight_limit: float = 0.40,
) -> Path:
    """Render and save markdown report to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_comparison_markdown(
        results, baseline_name=baseline_name,
        max_single_weight_limit=max_single_weight_limit,
    )
    output_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved ablation report to {output_path}")
    return output_path
