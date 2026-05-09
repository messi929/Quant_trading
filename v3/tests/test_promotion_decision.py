"""V3.3 Promotion decision tests (PR-5.2).

Verifies:
  - PromotionStep / PromotionPlan frozen
  - build_promotion_plan: 합격 ablation → ordered steps
  - 진단은 항상 week 0
  - 실패한 cumulative은 rejected_features에 포함
  - 누락된 ablation 처리
  - markdown rendering smoke
  - synthetic CLI integration smoke
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from v3.research.ablation_runner import AblationMetrics, ActionDistribution
from v3.research.promotion_decision import (
    CUMULATIVE_FEATURE_GROUPS,
    DIAGNOSTIC_FEATURES,
    PromotionPlan,
    PromotionStep,
    build_promotion_plan,
    render_promotion_markdown,
    save_promotion_plan,
)


def _clean_metrics(name: str, family: str, **overrides) -> AblationMetrics:
    """Helper: passing metrics."""
    defaults = dict(
        ablation_name=name,
        family=family,
        n_dates=20,
        n_actions=20,
        action_distribution=ActionDistribution(add_new=15, add_to_winner=2,
                                                keep=2, exit=1),
        total_target_weight=5.0,
        avg_deployed=0.50,
        max_single_weight=0.30,
        leverage_violations=0,
        pyramid_count=2,
        rotation_count=0,
        risk_exit_count=1,
    )
    defaults.update(overrides)
    return AblationMetrics(**defaults)


def _failing_metrics(name: str, family: str = "cumulative") -> AblationMetrics:
    """Helper: leverage_violations > 0 (will fail)."""
    return _clean_metrics(name, family, leverage_violations=2)


# ──────────────────────────────────────────────────────────────
# Frozen
# ──────────────────────────────────────────────────────────────
class TestFrozen:
    def test_promotion_step_frozen(self):
        step = PromotionStep(
            week=1, feature_group="test",
            feature_flags=("foo",),
            rationale="bar",
            rollback_signal="baz",
        )
        with pytest.raises(FrozenInstanceError):
            step.week = 99  # type: ignore[misc]

    def test_promotion_plan_frozen(self):
        plan = PromotionPlan(steps=())
        with pytest.raises(FrozenInstanceError):
            plan.steps = (1,)  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
class TestConstants:
    def test_cumulative_groups_match_a1_a9(self):
        # A1 ~ A9 = 9 keys
        assert len(CUMULATIVE_FEATURE_GROUPS) == 9
        assert "A1_calibrator" in CUMULATIVE_FEATURE_GROUPS
        assert "A9_rotation" in CUMULATIVE_FEATURE_GROUPS

    def test_diagnostic_features_3(self):
        assert set(DIAGNOSTIC_FEATURES) == {
            "no_trade_logger", "tc_monitor", "execution_quality",
        }


# ──────────────────────────────────────────────────────────────
# build_promotion_plan
# ──────────────────────────────────────────────────────────────
class TestBuildPromotionPlan:
    def test_diagnostics_always_week_zero(self):
        results = [
            _clean_metrics("baseline", "baseline"),
            _clean_metrics("A1_calibrator", "cumulative"),
        ]
        plan = build_promotion_plan(results)
        # First step is diagnostics
        assert plan.steps[0].week == 0
        assert plan.steps[0].feature_group == "diagnostics"
        assert set(plan.steps[0].feature_flags) == set(DIAGNOSTIC_FEATURES)

    def test_passing_cumulative_in_order(self):
        results = [
            _clean_metrics("baseline", "baseline"),
            _clean_metrics("A1_calibrator", "cumulative"),
            _clean_metrics("A2_edge_engine", "cumulative"),
            _clean_metrics("A3_tier", "cumulative"),
        ]
        plan = build_promotion_plan(results)
        # week 0 = diagnostics, then A1 (week 1), A2 (week 2), A3 (week 3)
        weeks = [(s.week, s.feature_group) for s in plan.steps]
        assert weeks == [
            (0, "diagnostics"),
            (1, "A1_calibrator"),
            (2, "A2_edge_engine"),
            (3, "A3_tier"),
        ]

    def test_failing_cumulative_added_to_rejected(self):
        results = [
            _clean_metrics("baseline", "baseline"),
            _clean_metrics("A1_calibrator", "cumulative"),
            _failing_metrics("A2_edge_engine"),  # FAIL
            _clean_metrics("A3_tier", "cumulative"),
        ]
        plan = build_promotion_plan(results)
        # A2 features in rejected
        assert "edge_engine" in plan.rejected_features
        # A1 and A3 still promoted (rejection doesn't block subsequent)
        names = [s.feature_group for s in plan.steps]
        assert "A1_calibrator" in names
        assert "A3_tier" in names
        assert "A2_edge_engine" not in names

    def test_missing_ablation_added_to_rejected_with_note(self):
        # Only baseline and A2 — A1, A3..A9 missing
        results = [
            _clean_metrics("baseline", "baseline"),
            _clean_metrics("A2_edge_engine", "cumulative"),
        ]
        plan = build_promotion_plan(results)
        assert "edge_calibrator" in plan.rejected_features
        assert any("A1_calibrator" in n for n in plan.notes)

    def test_full_pass_promotes_all(self):
        results = [_clean_metrics("baseline", "baseline")]
        for name in CUMULATIVE_FEATURE_GROUPS:
            results.append(_clean_metrics(name, "cumulative"))
        plan = build_promotion_plan(results)
        # 9 cumulative + 1 diagnostics = 10 steps
        assert len(plan.steps) == 10
        assert len(plan.rejected_features) == 0


# ──────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────
class TestMarkdown:
    def test_smoke(self):
        results = [_clean_metrics("baseline", "baseline")]
        plan = build_promotion_plan(results)
        md = render_promotion_markdown(plan)
        assert "Paper Promotion Plan" in md
        assert "diagnostics" in md
        assert "Week 0" in md or "week" in md.lower()

    def test_rejected_section_appears(self):
        results = [
            _clean_metrics("baseline", "baseline"),
            _failing_metrics("A1_calibrator"),
        ]
        plan = build_promotion_plan(results)
        md = render_promotion_markdown(plan)
        assert "Rejected" in md
        assert "edge_calibrator" in md

    def test_save_writes_file(self, tmp_path):
        results = [_clean_metrics("baseline", "baseline")]
        plan = build_promotion_plan(results)
        out = tmp_path / "plan.md"
        save_promotion_plan(plan, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Promotion" in content


# ──────────────────────────────────────────────────────────────
# CLI integration (synthetic)
# ──────────────────────────────────────────────────────────────
class TestCLISyntheticSmoke:
    def test_synthetic_sweep_produces_reports(self, tmp_path):
        from v3.scripts.run_ablation_sweep import main

        rc = main([
            "--output-dir", str(tmp_path),
            "--tag", "test",
            "--synthetic",
            "--n-dates", "5",
        ])
        assert rc == 0
        assert (tmp_path / "ablation_test_comparison.md").exists()
        assert (tmp_path / "ablation_test_promotion.md").exists()

    def test_production_mode_returns_error(self, tmp_path):
        from v3.scripts.run_ablation_sweep import main

        rc = main([
            "--output-dir", str(tmp_path),
            "--tag", "test",
            # No --synthetic → production mode (not wired)
        ])
        assert rc == 1
