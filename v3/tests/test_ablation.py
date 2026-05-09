"""V3.3 Ablation infrastructure tests (PR-5.1).

Verifies:
  - ABLATION_SPECS contains 16 specs (baseline + A1~A9 + Full + Full-NoDiag + 4 isolated)
  - get_spec / list_ablations
  - YAML save/load round-trip
  - Cumulative monotonicity (A1 ⊂ A2 ⊂ ... ⊂ A9)
  - AblationRunner on synthetic input → AblationMetrics
  - Comparison matrix DataFrame
  - Pass/fail evaluation (leverage invariant)
  - Markdown report smoke
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from v3.config.schema import FeatureFlagsConfig
from v3.research.ablation_runner import (
    AblationMetrics,
    AblationRunner,
    ActionDistribution,
    run_sweep,
)
from v3.research.ablation_report import (
    build_comparison_matrix,
    evaluate_ablation_pass,
    render_comparison_markdown,
    save_comparison_report,
)
from v3.research.ablation_specs import (
    ABLATION_SPECS,
    AblationSpec,
    get_spec,
    list_ablations,
    load_features_from_yaml,
    save_specs_as_yaml,
    specs_summary,
)
from v3.rules.entry import OperationalState
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.regime_v2 import Regime
from v3.strategy.signal import TradeSignal


# ──────────────────────────────────────────────────────────────
# AblationSpec registry
# ──────────────────────────────────────────────────────────────
class TestAblationSpecs:
    def test_16_specs_total(self):
        assert len(ABLATION_SPECS) == 16

    def test_unique_names(self):
        names = [s.name for s in ABLATION_SPECS]
        assert len(names) == len(set(names))

    def test_baseline_all_off(self):
        baseline = get_spec("baseline")
        flags = baseline.features.model_dump()
        assert not any(flags.values())

    def test_full_all_on(self):
        full = get_spec("full")
        flags = full.features.model_dump()
        assert all(flags.values())

    def test_full_nodiag_diagnostics_off(self):
        nd = get_spec("full_nodiag")
        assert not nd.features.no_trade_logger
        assert not nd.features.tc_monitor
        assert not nd.features.execution_quality
        # but decision-affecting flags ON
        assert nd.features.edge_calibrator
        assert nd.features.allocation

    def test_get_spec_unknown_raises(self):
        with pytest.raises(KeyError):
            get_spec("nonexistent")

    def test_list_ablations_filters_by_family(self):
        cumulative = list_ablations(family="cumulative")
        # A1~A9 = 9 cumulative
        assert len(cumulative) == 9

        isolated = list_ablations(family="isolated")
        assert len(isolated) == 4

        baseline = list_ablations(family="baseline")
        assert baseline == ["baseline"]

        full_family = list_ablations(family="full")
        assert len(full_family) == 2  # full + full_nodiag


# ──────────────────────────────────────────────────────────────
# Cumulative monotonicity (A1 ⊂ A2 ⊂ ...)
# ──────────────────────────────────────────────────────────────
class TestCumulativeMonotonicity:
    """Each cumulative ablation activates a strict superset of the previous."""

    def test_each_step_adds_features(self):
        cumulative = ["A1_calibrator", "A2_edge_engine", "A3_tier",
                      "A4_conditional_veto", "A5_exit_thesis",
                      "A6_decay_partial", "A7_allocation",
                      "A8_pyramid", "A9_rotation"]
        prev_active: set[str] = set()
        for name in cumulative:
            spec = get_spec(name)
            current_active = {
                k for k, v in spec.features.model_dump().items() if v
            }
            # Each step is strict superset
            assert prev_active.issubset(current_active), (
                f"{name} not superset of previous: "
                f"missing {prev_active - current_active}"
            )
            prev_active = current_active

    def test_full_equals_a9(self):
        full_active = {
            k for k, v in get_spec("full").features.model_dump().items() if v
        }
        a9_active = {
            k for k, v in get_spec("A9_rotation").features.model_dump().items() if v
        }
        assert full_active == a9_active


# ──────────────────────────────────────────────────────────────
# YAML I/O
# ──────────────────────────────────────────────────────────────
class TestYAMLPersistence:
    def test_save_and_reload_round_trip(self, tmp_path):
        paths = save_specs_as_yaml(tmp_path)
        assert len(paths) == 16

        # Reload baseline
        baseline_path = tmp_path / "baseline.yaml"
        assert baseline_path.exists()
        flags = load_features_from_yaml(baseline_path)
        assert flags.model_dump() == get_spec("baseline").features.model_dump()

    def test_reload_full(self, tmp_path):
        save_specs_as_yaml(tmp_path)
        flags = load_features_from_yaml(tmp_path / "full.yaml")
        assert all(flags.model_dump().values())


# ──────────────────────────────────────────────────────────────
# specs_summary
# ──────────────────────────────────────────────────────────────
class TestSpecsSummary:
    def test_baseline_zero_active(self):
        s = specs_summary()
        assert s["baseline"]["n_active"] == 0

    def test_full_all_13_active(self):
        s = specs_summary()
        assert s["full"]["n_active"] == 13

    def test_a1_has_4_active(self):
        # diagnostics 3 + edge_calibrator 1
        s = specs_summary()
        assert s["A1_calibrator"]["n_active"] == 4


# ──────────────────────────────────────────────────────────────
# AblationRunner — synthetic execution
# ──────────────────────────────────────────────────────────────
def _mock_signal(action: str = "TRADE") -> TradeSignal:
    return TradeSignal(
        action=action,
        positions=[
            {"ticker": "AAPL", "weight": 0.30, "direction": 0.05,
             "conviction": 0.7, "opportunity": 0.0035},
        ] if action == "TRADE" else [],
        cash_weight=0.70 if action == "TRADE" else 1.0,
        regime_name="neutral",
        regime_score=0.5,
        position_scale=0.9,
        n_candidates=5,
        n_approved=1 if action == "TRADE" else 0,
        rejection_reasons={"net_edge": 4},
        opportunity_map={"AAPL": 0.0035, "MSFT": 0.0010, "GOOG": 0.0005},
        opportunity_gate=0.00175,
    )


def _mock_signal_generator(signal: TradeSignal) -> MagicMock:
    sg = MagicMock()
    sg.generate.return_value = signal
    return sg


def _regime() -> Regime:
    return Regime(
        name="neutral", score=0.5,
        alpha_weights={"trend": 1.0, "reversion": 0.0},
        position_scale=0.9, confidence=1.0,
        detected_at=pd.Timestamp("2026-05-09"),
    )


def _state() -> OperationalState:
    return OperationalState(
        current_positions=0,
        monthly_trades=0,
        recent_win_rate=0.5,
        current_mdd=0.0,
        circuit_breaker_active=False,
    )


def _make_factory():
    """Factory creating a BookOptimizer with mocked SignalGenerator only.

    Intentionally minimal — Phase 5.2에서 진짜 의존성 주입 통합.
    Diagnostic / Edge / Phase 3+4 deps are None — features.* OFF means
    ablations that need them effectively passthrough.
    """
    signal = _mock_signal(action="TRADE")
    sg = _mock_signal_generator(signal)

    def factory(features: FeatureFlagsConfig) -> BookOptimizer:
        return BookOptimizer(signal_gen=sg, features=features)

    return factory


def _make_inputs(n_dates: int = 5):
    """Yield n synthetic input dicts."""
    for i in range(n_dates):
        yield {
            "date": pd.Timestamp("2026-05-09") + pd.Timedelta(days=i),
            "ohlcv": pd.DataFrame(),
            "vol_scores": pd.DataFrame(),
            "regime": _regime(),
            "cost": 0.001,
            "state": _state(),
        }


class TestAblationRunner:
    def test_baseline_runs_and_returns_metrics(self):
        runner = AblationRunner(book_optimizer_factory=_make_factory())
        spec = get_spec("baseline")
        metrics = runner.run(spec, _make_inputs(n_dates=3))
        assert isinstance(metrics, AblationMetrics)
        assert metrics.n_dates == 3
        assert metrics.ablation_name == "baseline"
        assert metrics.n_actions > 0

    def test_action_distribution_total_matches(self):
        runner = AblationRunner(book_optimizer_factory=_make_factory())
        spec = get_spec("baseline")
        metrics = runner.run(spec, _make_inputs(n_dates=5))
        assert metrics.action_distribution.total == metrics.n_actions

    def test_baseline_passthrough_no_pyramid(self):
        runner = AblationRunner(book_optimizer_factory=_make_factory())
        spec = get_spec("baseline")
        metrics = runner.run(spec, _make_inputs(n_dates=3))
        # All flags OFF → no Phase 4 pyramid action
        assert metrics.pyramid_count == 0
        assert metrics.rotation_count == 0


# ──────────────────────────────────────────────────────────────
# run_sweep
# ──────────────────────────────────────────────────────────────
class TestRunSweep:
    def test_sweep_produces_per_spec_metrics(self):
        runner = AblationRunner(book_optimizer_factory=_make_factory())
        specs = [get_spec("baseline"), get_spec("A1_calibrator")]
        results = run_sweep(
            specs,
            runner=runner,
            inputs_factory=lambda: _make_inputs(n_dates=2),
        )
        assert len(results) == 2
        assert results[0].ablation_name == "baseline"
        assert results[1].ablation_name == "A1_calibrator"


# ──────────────────────────────────────────────────────────────
# Comparison matrix + report
# ──────────────────────────────────────────────────────────────
def _sample_results() -> list[AblationMetrics]:
    return [
        AblationMetrics(
            ablation_name="baseline", family="baseline",
            n_dates=10, n_actions=10,
            action_distribution=ActionDistribution(add_new=10),
            total_target_weight=2.5, avg_deployed=0.25,
            max_single_weight=0.30, leverage_violations=0,
            pyramid_count=0, rotation_count=0, risk_exit_count=0,
        ),
        AblationMetrics(
            ablation_name="full", family="full",
            n_dates=10, n_actions=15,
            action_distribution=ActionDistribution(
                add_new=8, add_to_winner=4, keep=2, exit=1,
            ),
            total_target_weight=4.5, avg_deployed=0.45,
            max_single_weight=0.38, leverage_violations=0,
            pyramid_count=4, rotation_count=0, risk_exit_count=1,
        ),
    ]


class TestComparisonMatrix:
    def test_dataframe_one_row_per_ablation(self):
        df = build_comparison_matrix(_sample_results())
        assert len(df) == 2
        assert "ablation" in df.columns
        assert "leverage_violations" in df.columns
        assert "pyramid_count" in df.columns

    def test_metrics_correctly_extracted(self):
        df = build_comparison_matrix(_sample_results())
        full_row = df[df["ablation"] == "full"].iloc[0]
        assert full_row["pyramid_count"] == 4
        assert full_row["leverage_violations"] == 0
        assert full_row["avg_deployed"] == pytest.approx(0.45)


# ──────────────────────────────────────────────────────────────
# Pass/fail evaluation
# ──────────────────────────────────────────────────────────────
class TestPassFailEvaluation:
    def test_clean_metrics_pass(self):
        m = _sample_results()[0]  # baseline, no violations
        passed, failures = evaluate_ablation_pass(m)
        assert passed
        assert failures == []

    def test_leverage_violation_fails(self):
        m = AblationMetrics(
            ablation_name="bad", family="cumulative",
            n_dates=10, n_actions=5,
            action_distribution=ActionDistribution(add_new=5),
            total_target_weight=5.0, avg_deployed=0.5,
            max_single_weight=0.30, leverage_violations=2,  # !!!
            pyramid_count=0, rotation_count=0, risk_exit_count=0,
        )
        passed, failures = evaluate_ablation_pass(m)
        assert not passed
        assert any("leverage_violations" in f for f in failures)

    def test_max_single_weight_violation(self):
        m = AblationMetrics(
            ablation_name="bad", family="cumulative",
            n_dates=10, n_actions=5,
            action_distribution=ActionDistribution(add_new=5),
            total_target_weight=2.5, avg_deployed=0.25,
            max_single_weight=0.45,  # > 0.40
            leverage_violations=0,
            pyramid_count=0, rotation_count=0, risk_exit_count=0,
        )
        passed, failures = evaluate_ablation_pass(m, max_single_weight_limit=0.40)
        assert not passed
        assert any("max_single_weight" in f for f in failures)

    def test_pyramid_below_baseline_fails(self):
        baseline = AblationMetrics(
            ablation_name="baseline", family="baseline",
            n_dates=10, n_actions=10,
            action_distribution=ActionDistribution(add_new=8, add_to_winner=2),
            total_target_weight=2.5, avg_deployed=0.25,
            max_single_weight=0.30, leverage_violations=0,
            pyramid_count=2, rotation_count=0, risk_exit_count=0,
        )
        target = AblationMetrics(
            ablation_name="A8_pyramid", family="cumulative",
            n_dates=10, n_actions=10,
            action_distribution=ActionDistribution(add_new=8, add_to_winner=0),
            total_target_weight=2.5, avg_deployed=0.25,
            max_single_weight=0.30, leverage_violations=0,
            pyramid_count=0,  # < baseline 2
            rotation_count=0, risk_exit_count=0,
        )
        passed, failures = evaluate_ablation_pass(target, baseline=baseline)
        assert not passed
        assert any("pyramid count" in f for f in failures)


# ──────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────
class TestMarkdownReport:
    def test_smoke(self):
        md = render_comparison_markdown(_sample_results())
        assert "# V3.3 Ablation Comparison Report" in md
        assert "Comparison Matrix" in md
        assert "Pass/Fail Summary" in md
        assert "baseline" in md
        assert "full" in md

    def test_save_writes_file(self, tmp_path):
        out = tmp_path / "report.md"
        path = save_comparison_report(_sample_results(), out)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Ablation Comparison" in content
