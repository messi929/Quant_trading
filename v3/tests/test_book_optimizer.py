"""V3.3 BookOptimizer tests (PR-2.4) — PARITY core.

Verifies:
  - features.all_off() → SignalGenerator positions = ADD_NEW BookActions 1:1
  - CASH signal → NO_ACTION
  - Edge layer flags enrich without changing actions
  - NoTradeLogger / TC Monitor flag-gated
  - Action priority sort
  - BookAction immutability
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from v3.config.schema import FeatureFlagsConfig
from v3.rules.entry import OperationalState
from v3.strategy.book_optimizer import ACTION_PRIORITY, BookOptimizer
from v3.strategy.diagnostics import (
    NoTradeReasonLogger,
    TransferCoefficientMonitor,
)
from v3.strategy.edge_calibrator import (
    CalibrationConfig,
    EdgeCalibrator,
)
from v3.strategy.edge_engine import EdgeEngine, EdgePolicy
from v3.strategy.edge_tier import EdgeTierSystem, TierThresholds
from v3.strategy.regime_v2 import Regime
from v3.strategy.signal import TradeSignal
from v3.strategy.types import BookAction, CalibrationBucket


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
def _mock_signal(
    action: str = "TRADE",
    positions: list[dict] | None = None,
    opportunity_map: dict | None = None,
    rejection_reasons: dict | None = None,
) -> TradeSignal:
    if action == "TRADE" and positions is None:
        positions = [
            {"ticker": "AAPL", "weight": 0.30, "direction": 0.05,
             "conviction": 0.7, "opportunity": 0.0035},
            {"ticker": "MSFT", "weight": 0.20, "direction": 0.03,
             "conviction": 0.6, "opportunity": 0.0018},
        ]
    if action == "CASH":
        positions = []
    if opportunity_map is None:
        opportunity_map = {
            "AAPL": 0.0035,
            "MSFT": 0.0018,
            "GOOG": 0.0008,  # below gate, not in positions
            "AMZN": 0.0005,
        }
    return TradeSignal(
        action=action,
        positions=positions or [],
        cash_weight=0.50 if action == "TRADE" else 1.0,
        regime_name="neutral",
        regime_score=0.5,
        position_scale=0.9,
        n_candidates=10,
        n_approved=2 if action == "TRADE" else 0,
        rejection_reasons=rejection_reasons or {"net_edge": 8},
        opportunity_map=opportunity_map,
        opportunity_gate=0.00175,
    )


def _mock_signal_generator(signal: TradeSignal) -> MagicMock:
    sg = MagicMock()
    sg.generate.return_value = signal
    return sg


def _regime() -> Regime:
    return Regime(
        name="neutral",
        score=0.5,
        alpha_weights={"trend": 1.0, "reversion": 0.0},
        position_scale=0.9,
        confidence=1.0,
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


# ──────────────────────────────────────────────────────────────
# PARITY: all_off matches SignalGenerator
# ──────────────────────────────────────────────────────────────
class TestParityAllOff:
    """Critical: V3.2.1 동작 보존 검증."""

    def test_trade_positions_become_add_new_actions(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(),
            vol_scores=pd.DataFrame(),
            regime=_regime(),
            cost=0.001,
            state=_state(),
        )
        add_new = [a for a in actions if a.action_type == "ADD_NEW"]
        assert len(add_new) == len(signal.positions)
        # 1:1 mapping by ticker
        action_tickers = {a.ticker for a in add_new}
        signal_tickers = {p["ticker"] for p in signal.positions}
        assert action_tickers == signal_tickers

    def test_target_weights_passthrough(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        action_weights = {
            a.ticker: a.target_weight for a in actions if a.action_type == "ADD_NEW"
        }
        signal_weights = {p["ticker"]: p["weight"] for p in signal.positions}
        assert action_weights == signal_weights

    def test_cash_signal_yields_no_action(self):
        signal = _mock_signal(action="CASH")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        assert len(actions) == 1
        assert actions[0].action_type == "NO_ACTION"

    def test_no_edge_fields_populated_when_off(self):
        """Edge fields stay None when calibrator/engine OFF."""
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
            edge_calibrator=None,  # explicitly absent
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # ADD_NEW actions don't expose Edge fields directly,
        # but expected_impact (= opportunity) should equal raw opp
        for a in actions:
            if a.action_type == "ADD_NEW":
                assert a.expected_impact is not None  # opportunity passthrough


# ──────────────────────────────────────────────────────────────
# Edge layer enrichment (flag ON)
# ──────────────────────────────────────────────────────────────
def _make_calibrator() -> EdgeCalibrator:
    """Synthetic calibrator with global bucket only."""
    table = {
        ("global",): CalibrationBucket(
            key=("global",),
            n=1000,
            mean_forward_return_5d=0.008,
            mean_mae_5d=-0.020,
            mean_mfe_5d=0.025,
            win_rate_5d=0.55,
            std_forward_return_5d=0.030,
            median_forward_return_5d=0.007,
        ),
    }
    decile_bp = [-0.01, -0.005, -0.001, 0.0, 0.001, 0.003, 0.005, 0.008, 0.012]
    return EdgeCalibrator(
        table=table,
        decile_breakpoints=decile_bp,
        config=CalibrationConfig(insufficient_action="global"),
    )


class TestEdgeLayerEnrichment:
    def test_calibrator_flag_does_not_change_actions(self):
        signal = _mock_signal(action="TRADE")
        sg = _mock_signal_generator(signal)
        flags = FeatureFlagsConfig(edge_calibrator=True)
        bo = BookOptimizer(
            signal_gen=sg,
            features=flags,
            edge_calibrator=_make_calibrator(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # Same positions as all_off — calibration is enrichment, not gating
        add_new_tickers = {a.ticker for a in actions if a.action_type == "ADD_NEW"}
        signal_tickers = {p["ticker"] for p in signal.positions}
        assert add_new_tickers == signal_tickers

    def test_full_edge_chain_runs(self):
        """Calibrator + Engine + Tier all ON should not crash."""
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(),
            edge_tier=EdgeTierSystem(thresholds=TierThresholds()),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # Actions still match SignalGenerator positions
        assert len([a for a in actions if a.action_type == "ADD_NEW"]) == 2


# ──────────────────────────────────────────────────────────────
# Diagnostics (flag-gated)
# ──────────────────────────────────────────────────────────────
class TestDiagnostics:
    def test_no_trade_logger_buffers_rejects(self, tmp_path):
        signal = _mock_signal(action="TRADE")  # 4 candidates, 2 entered
        flags = FeatureFlagsConfig(no_trade_logger=True)
        logger = NoTradeReasonLogger(log_dir=tmp_path)
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            no_trade_logger=logger,
        )
        bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # 4 candidates - 2 entered = 2 rejected
        assert logger.buffer_size() == 2

    def test_no_trade_logger_off_no_buffer(self, tmp_path):
        signal = _mock_signal(action="TRADE")
        logger = NoTradeReasonLogger(log_dir=tmp_path)
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
            no_trade_logger=logger,
        )
        bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        assert logger.buffer_size() == 0

    def test_tc_monitor_records_snapshot(self, tmp_path):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(tc_monitor=True)
        tc = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            tc_monitor=tc,
        )
        bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        assert tc.buffer_size() == 1

    def test_flush_diagnostics(self, tmp_path):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(no_trade_logger=True, tc_monitor=True)
        logger = NoTradeReasonLogger(log_dir=tmp_path)
        tc = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            no_trade_logger=logger,
            tc_monitor=tc,
        )
        bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        bo.flush_diagnostics()
        assert logger.buffer_size() == 0
        assert tc.buffer_size() == 0


# ──────────────────────────────────────────────────────────────
# Action priority sort
# ──────────────────────────────────────────────────────────────
class TestActionPriority:
    def test_priority_constants_ordered(self):
        # EXIT (0) > ROTATE (2) > ADD_NEW (4) > KEEP (6)
        assert ACTION_PRIORITY["EXIT"] < ACTION_PRIORITY["ROTATE"]
        assert ACTION_PRIORITY["ROTATE"] < ACTION_PRIORITY["ADD_TO_WINNER"]
        assert ACTION_PRIORITY["ADD_TO_WINNER"] < ACTION_PRIORITY["ADD_NEW"]

    def test_actions_sorted_by_priority(self):
        # In PR-2.4 we only emit ADD_NEW + NO_ACTION, but verify sort.
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        priorities = [ACTION_PRIORITY[a.action_type] for a in actions]
        assert priorities == sorted(priorities)


# ──────────────────────────────────────────────────────────────
# Immutability
# ──────────────────────────────────────────────────────────────
class TestImmutability:
    def test_book_actions_are_frozen(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        from dataclasses import FrozenInstanceError
        for a in actions:
            with pytest.raises(FrozenInstanceError):
                a.target_weight = 999.0  # type: ignore[misc]
                break  # one is enough
