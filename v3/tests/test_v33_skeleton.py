"""V3.3 skeleton tests (PR-1.0).

Verifies:
  - Frozen invariants on all V3.3 dataclasses
  - FeatureFlagsConfig defaults all OFF
  - Protocol runtime_checkable
  - load_config() includes features section

This file is a foundation — subsequent PRs add tests for each module
(test_diagnostics.py, test_edge_calibrator.py, etc.).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from v3.config.schema import FeatureFlagsConfig, load_config
from v3.strategy._base import DecisionPolicy, DiagnosticSink
from v3.strategy.types import (
    BookAction,
    CalibrationBucket,
    EdgeCandidate,
    NoTradeLog,
    PositionState,
)


# ──────────────────────────────────────────────────────────────
# Frozen dataclass invariants
# ──────────────────────────────────────────────────────────────
class TestFrozenDataclasses:
    def _make_candidate(self) -> EdgeCandidate:
        return EdgeCandidate(
            date=pd.Timestamp("2026-05-09"),
            ticker="AAPL",
            direction=0.05,
            conviction=0.7,
            raw_opportunity=0.035,
            regime="neutral",
            regime_score=0.5,
        )

    def test_edge_candidate_frozen(self):
        c = self._make_candidate()
        with pytest.raises(FrozenInstanceError):
            c.direction = 0.10  # type: ignore[misc]

    def test_edge_candidate_with_calibration_returns_new(self):
        c = self._make_candidate()
        c2 = c.with_calibration(expected_return_5d=0.012, calibration_n=150)
        assert c.expected_return_5d is None
        assert c2.expected_return_5d == pytest.approx(0.012)
        assert c2.calibration_n == 150
        assert c2 is not c
        # Original untouched
        assert c.calibration_n is None

    def test_book_action_frozen(self):
        a = BookAction(
            date=pd.Timestamp("2026-05-09"),
            action_type="ADD_NEW",
            ticker="AAPL",
            target_weight=0.20,
            current_weight=0.0,
            reason="test entry",
            source_policy="test",
        )
        with pytest.raises(FrozenInstanceError):
            a.target_weight = 0.30  # type: ignore[misc]

    def test_book_action_weight_delta(self):
        a = BookAction(
            date=pd.Timestamp("2026-05-09"),
            action_type="ADD_TO_WINNER",
            ticker="AAPL",
            target_weight=0.30,
            current_weight=0.20,
            reason="winner add",
            source_policy="PyramidPolicy",
        )
        assert a.weight_delta == pytest.approx(0.10)

    def test_book_action_is_risk_exit(self):
        risk = BookAction(
            date=pd.Timestamp("2026-05-09"),
            action_type="EXIT",
            ticker="AAPL",
            target_weight=0.0,
            current_weight=0.20,
            reason="risk_exit:vol_contraction",
            source_policy="ExitThesisEngine",
        )
        assert risk.is_risk_exit

        normal_exit = BookAction(
            date=pd.Timestamp("2026-05-09"),
            action_type="EXIT",
            ticker="AAPL",
            target_weight=0.0,
            current_weight=0.20,
            reason="residual_edge_negative",
            source_policy="ExitThesisEngine",
        )
        assert not normal_exit.is_risk_exit

    def test_position_state_frozen(self):
        p = PositionState(
            ticker="AAPL",
            entry_date=pd.Timestamp("2026-05-01"),
            entry_price=200.0,
            current_price=210.0,
            current_weight=0.20,
            entry_edge=0.008,
            residual_edge=0.005,
            entry_tier="A",
            current_tier="A",
            pnl_pct=0.05,
            max_unrealized_pnl_pct=0.06,
            drawdown_from_peak_pct=-0.01,
            holding_days=8,
            dominant_alpha="trend",
            expected_holding_days=10,
            thesis_alive=True,
        )
        with pytest.raises(FrozenInstanceError):
            p.current_price = 220.0  # type: ignore[misc]

    def test_no_trade_log_frozen(self):
        log = NoTradeLog(
            date=pd.Timestamp("2026-05-09"),
            ticker="AAPL",
            stage="edge",
            reject_reason="net_edge_low",
            raw_opportunity=0.002,
            regime="caution",
        )
        with pytest.raises(FrozenInstanceError):
            log.reject_reason = "other"  # type: ignore[misc]


class TestCalibrationBucket:
    def _make_bucket(self, n: int = 50) -> CalibrationBucket:
        return CalibrationBucket(
            key=("global",),
            n=n,
            mean_forward_return_5d=0.012,
            mean_mae_5d=-0.020,
            mean_mfe_5d=0.025,
            win_rate_5d=0.55,
            std_forward_return_5d=0.030,
            median_forward_return_5d=0.010,
        )

    def test_is_reliable_default_threshold(self):
        b = self._make_bucket(n=50)
        assert b.is_reliable(min_n=30)
        assert not b.is_reliable(min_n=100)

    def test_standard_error_finite(self):
        b = self._make_bucket(n=100)
        # SE = std / sqrt(n) = 0.030 / 10 = 0.003
        assert b.standard_error == pytest.approx(0.003)

    def test_standard_error_zero_sample(self):
        b = CalibrationBucket(
            key=("empty",), n=0,
            mean_forward_return_5d=0.0, mean_mae_5d=0.0, mean_mfe_5d=0.0,
            win_rate_5d=0.0, std_forward_return_5d=0.0,
            median_forward_return_5d=0.0,
        )
        assert b.standard_error == float("inf")

    def test_frozen(self):
        b = self._make_bucket()
        with pytest.raises(FrozenInstanceError):
            b.n = 999  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────
# FeatureFlagsConfig
# ──────────────────────────────────────────────────────────────
class TestFeatureFlagsConfig:
    def test_all_off_default(self):
        f = FeatureFlagsConfig.all_off()
        # All 12 flags False (conditional_veto removed in F1 fix)
        flags = [
            f.no_trade_logger, f.tc_monitor, f.execution_quality,
            f.edge_calibrator, f.edge_engine, f.edge_tier, f.allocation,
            f.exit_thesis, f.partial_exit, f.signal_decay,
            f.pyramid, f.rotation,
        ]
        assert not any(flags)
        assert len(flags) == 12

    def test_default_constructor_is_all_off(self):
        # Default constructor must equal all_off()
        f1 = FeatureFlagsConfig()
        f2 = FeatureFlagsConfig.all_off()
        assert f1.model_dump() == f2.model_dump()

    def test_load_config_has_features(self):
        cfg = load_config()
        assert hasattr(cfg, "features")
        assert isinstance(cfg.features, FeatureFlagsConfig)
        # Production config must have all flags OFF currently
        # (V3.3 features 활성화는 ROADMAP §5 paper promotion phase에서)
        for flag_name in cfg.features.model_dump():
            assert cfg.features.model_dump()[flag_name] is False, (
                f"Feature {flag_name} must be OFF in v3_config.yaml until "
                f"paper promotion phase (see docs/V3.3_ROADMAP.md §5)"
            )


# ──────────────────────────────────────────────────────────────
# Protocols (runtime_checkable)
# ──────────────────────────────────────────────────────────────
class TestProtocols:
    def test_diagnostic_sink_runtime_checkable(self):
        class Dummy:
            name = "dummy"
            def record(self, event: dict) -> None:
                pass
            def flush(self) -> None:
                pass

        d = Dummy()
        assert isinstance(d, DiagnosticSink)

    def test_diagnostic_sink_rejects_missing_methods(self):
        class Incomplete:
            name = "incomplete"
            # Missing record() and flush()

        d = Incomplete()
        assert not isinstance(d, DiagnosticSink)

    def test_decision_policy_runtime_checkable(self):
        class DummyPolicy:
            name = "dummy"
            @property
            def enabled(self) -> bool:
                return True

        p = DummyPolicy()
        assert isinstance(p, DecisionPolicy)
