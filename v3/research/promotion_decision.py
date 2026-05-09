"""V3.3 Paper promotion decision generator (PR-5.2).

After ablation sweep, decide which features to activate in paper trading
and in what order. Output is a markdown promotion plan.

Logic:
  1. Filter ablations that PASSED V3.3_AB_PLAN §6 criteria
  2. For cumulative family: identify highest-numbered passing step
     (A9 best > A8 next > ...). Activate features incrementally.
  3. For isolated family: identify standalone winners (passed + showed effect).
  4. Generate ordered activation plan with per-week schedule.

Output:
  - promotion_plan: ordered list of (feature_group, week_index, rationale)
  - markdown report

This module does NOT execute promotion. It only produces a recommendation
document for human review (V3.3_ROADMAP §5.4 paper promotion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from v3.research.ablation_report import evaluate_ablation_pass
from v3.research.ablation_runner import AblationMetrics


# ──────────────────────────────────────────────────────────────
# PromotionStep
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PromotionStep:
    """Single feature activation step in the paper promotion plan."""
    week: int
    feature_group: str
    feature_flags: tuple[str, ...]
    rationale: str
    rollback_signal: str  # condition to roll back if observed in paper


@dataclass(frozen=True)
class PromotionPlan:
    """Full paper promotion plan."""
    steps: tuple[PromotionStep, ...] = field(default_factory=tuple)
    rejected_features: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


# ──────────────────────────────────────────────────────────────
# Cumulative-step → feature group mapping
# ──────────────────────────────────────────────────────────────
# Each cumulative ablation activates exactly one new group on top of
# previous. We use this to derive activation order.
CUMULATIVE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A1_calibrator": ("edge_calibrator",),
    "A2_edge_engine": ("edge_engine",),
    "A3_tier": ("edge_tier",),
    "A4_conditional_veto": ("conditional_veto",),
    "A5_exit_thesis": ("exit_thesis",),
    "A6_decay_partial": ("signal_decay", "partial_exit"),
    "A7_allocation": ("allocation",),
    "A8_pyramid": ("pyramid",),
    "A9_rotation": ("rotation",),
}

# Diagnostics — always promotable immediately (read-only)
DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "no_trade_logger",
    "tc_monitor",
    "execution_quality",
)


# ──────────────────────────────────────────────────────────────
# Plan builder
# ──────────────────────────────────────────────────────────────
def build_promotion_plan(
    results: list[AblationMetrics],
    baseline_name: str = "baseline",
    max_single_weight_limit: float = 0.40,
) -> PromotionPlan:
    """Build paper promotion plan from ablation results.

    Args:
        results: AblationMetrics from sweep
        baseline_name: name of baseline ablation for comparison

    Returns:
        PromotionPlan with ordered steps + rejected features.
    """
    by_name = {m.ablation_name: m for m in results}
    baseline = by_name.get(baseline_name)
    notes: list[str] = []

    if baseline is None:
        notes.append(f"WARNING: no '{baseline_name}' result found — skipping baseline comparison")

    # Step 1: Diagnostics always go first (independent of ablation outcome)
    steps: list[PromotionStep] = []
    steps.append(PromotionStep(
        week=0,
        feature_group="diagnostics",
        feature_flags=DIAGNOSTIC_FEATURES,
        rationale=(
            "Read-only — no decision impact. Activate immediately to gather "
            "operational data (no_trade_logger, tc_monitor, execution_quality)."
        ),
        rollback_signal="N/A (read-only)",
    ))

    # Step 2: Cumulative passing steps — promote in order
    rejected: list[str] = []
    week_index = 1
    cumulative_order = list(CUMULATIVE_FEATURE_GROUPS.keys())
    for ablation_name in cumulative_order:
        m = by_name.get(ablation_name)
        if m is None:
            # Missing — treat features as rejected (consistent with FAIL handling)
            rejected.extend(CUMULATIVE_FEATURE_GROUPS[ablation_name])
            notes.append(f"{ablation_name}: result missing")
            continue

        passed, failures = evaluate_ablation_pass(
            m, baseline=baseline,
            max_single_weight_limit=max_single_weight_limit,
        )
        if not passed:
            rejected.extend(CUMULATIVE_FEATURE_GROUPS[ablation_name])
            notes.append(
                f"{ablation_name} FAILED: {'; '.join(failures)}"
            )
            continue

        steps.append(PromotionStep(
            week=week_index,
            feature_group=ablation_name,
            feature_flags=CUMULATIVE_FEATURE_GROUPS[ablation_name],
            rationale=(
                f"Ablation {m.ablation_name} passed all V3.3_AB_PLAN §6 criteria. "
                f"avg_deployed={m.avg_deployed:.3f}, "
                f"pyramid_count={m.pyramid_count}, "
                f"rotation_count={m.rotation_count}, "
                f"leverage_violations={m.leverage_violations}."
            ),
            rollback_signal=(
                "Deactivate if 1-week paper PnL is -2% worse than baseline OR "
                "leverage_violations > 0 OR risk_exit failure."
            ),
        ))
        week_index += 1

    return PromotionPlan(
        steps=tuple(steps),
        rejected_features=tuple(rejected),
        notes=tuple(notes),
    )


# ──────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────
def render_promotion_markdown(plan: PromotionPlan) -> str:
    """Render PromotionPlan as markdown."""
    lines: list[str] = []
    lines.append("# V3.3 Paper Promotion Plan")
    lines.append("")
    lines.append(f"- Total promotion steps: {len(plan.steps)}")
    lines.append(f"- Rejected features: {len(plan.rejected_features)}")
    lines.append("")

    if plan.steps:
        lines.append("## Activation Schedule")
        lines.append("")
        lines.append("| Week | Group | Features | Rollback signal |")
        lines.append("|---:|---|---|---|")
        for step in plan.steps:
            features_str = ", ".join(step.feature_flags)
            lines.append(
                f"| {step.week} | {step.feature_group} | "
                f"{features_str} | {step.rollback_signal} |"
            )
        lines.append("")

        lines.append("## Rationale per Step")
        lines.append("")
        for step in plan.steps:
            lines.append(f"### Week {step.week}: {step.feature_group}")
            lines.append("")
            lines.append(f"**Features to activate:** {', '.join(step.feature_flags)}")
            lines.append("")
            lines.append(f"**Rationale:** {step.rationale}")
            lines.append("")
            lines.append(f"**Rollback signal:** {step.rollback_signal}")
            lines.append("")

    if plan.rejected_features:
        lines.append("## Rejected Features (Not Promoted)")
        lines.append("")
        for feat in plan.rejected_features:
            lines.append(f"- `{feat}`")
        lines.append("")

    if plan.notes:
        lines.append("## Notes")
        lines.append("")
        for note in plan.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def save_promotion_plan(
    plan: PromotionPlan,
    output_path: str | Path,
) -> Path:
    """Save markdown promotion plan to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_promotion_markdown(plan)
    output_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved promotion plan to {output_path}")
    return output_path
