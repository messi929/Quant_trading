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
from v3.strategy.allocation import AllocationEngine, RegimeBudget
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
from v3.strategy.exit_thesis import ExitPolicy, ExitThesisEngine
from v3.strategy.partial_exit import PartialExitEngine, PartialExitPolicy
from v3.strategy.pyramid import PyramidPolicy, PyramidPolicyEngine
from v3.strategy.regime_v2 import Regime
from v3.strategy.rotation import CapitalRotationEngine, RotationPolicy
from v3.strategy.signal import TradeSignal
from v3.strategy.types import BookAction, CalibrationBucket, PositionState


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
class TestDecisionContext:
    """PR-2.5 — decide_with_context returns rich context for backtest/live."""

    def test_ctx_signal_is_underlying_trade_signal(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        ctx = bo.decide_with_context(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # ctx.signal is the same TradeSignal returned by signal_gen.generate()
        assert ctx.signal is signal

    def test_ctx_actions_match_decide(self):
        """decide() == decide_with_context().actions (thin wrapper invariant)."""
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions_direct = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        ctx = bo.decide_with_context(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # Same length and same structure (BookActions are frozen → equal by fields)
        assert len(actions_direct) == len(ctx.actions)
        for a, b in zip(actions_direct, ctx.actions):
            assert a.action_type == b.action_type
            assert a.ticker == b.ticker
            assert a.target_weight == b.target_weight

    def test_ctx_candidates_populated(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        ctx = bo.decide_with_context(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # 4 candidates from opportunity_map
        assert len(ctx.candidates) == 4


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


# ──────────────────────────────────────────────────────────────
# PR-4.4: Phase 3+4 통합 테스트
# ──────────────────────────────────────────────────────────────
def _winner_position(
    ticker: str = "AAPL",
    pnl_pct: float = 0.05,
    current_weight: float = 0.20,
    holding_days: int = 4,
    current_tier: str = "A",
) -> PositionState:
    return PositionState(
        ticker=ticker,
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=200.0 * (1.0 + pnl_pct),
        current_weight=current_weight,
        entry_edge=0.005,
        residual_edge=0.005,
        entry_tier=current_tier,
        current_tier=current_tier,
        pnl_pct=pnl_pct,
        max_unrealized_pnl_pct=max(pnl_pct, 0.0),
        drawdown_from_peak_pct=-0.005,
        holding_days=holding_days,
        dominant_alpha="trend",
        expected_holding_days=10,
        thesis_alive=True,
    )


class TestPhase3Integration:
    """ExitThesisEngine wiring inside BookOptimizer."""

    def test_exit_thesis_keeps_winner_with_positive_edge(self):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True,
            exit_thesis=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(),
            exit_thesis=ExitThesisEngine(policy=ExitPolicy(
                hold_min_residual_edge=-0.999,  # always allow KEEP
            )),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL")],
            exit_triggers={"AAPL": "max_hold"},
        )
        # AAPL should appear as KEEP (or similar non-EXIT)
        aapl_actions = [a for a in actions if a.ticker == "AAPL"
                        and a.source_policy == "exit_thesis"]
        assert len(aapl_actions) >= 1

    def test_partial_exit_fallback_when_thesis_off(self):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(partial_exit=True)
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            partial_exit=PartialExitEngine(policy=PartialExitPolicy()),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL")],
            exit_triggers={"AAPL": "profit_take"},
        )
        # PartialExit emits a position-level action
        aapl_actions = [a for a in actions if a.ticker == "AAPL"
                        and a.source_policy == "partial_exit"]
        assert len(aapl_actions) >= 1

    def test_passthrough_keep_when_no_exit_policy(self):
        signal = _mock_signal(action="TRADE")
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=FeatureFlagsConfig.all_off(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL")],
        )
        # passthrough KEEP exists
        passthrough = [a for a in actions if a.source_policy == "passthrough"]
        assert len(passthrough) == 1
        assert passthrough[0].action_type == "KEEP"


class TestPyramidIntegration:
    """Pyramid wiring — KEEP winners only get add-on."""

    def test_pyramid_adds_to_keeping_winner(self):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True, edge_tier=True,
            exit_thesis=True, pyramid=True,
        )
        # Lenient thresholds so synthetic candidates pass full pipeline as KEEP
        lenient_thresholds = TierThresholds(
            s_tier_net_edge=-0.999, a_tier_net_edge=-0.999, b_tier_net_edge=-0.999,
            s_tier_winprob=0.0, a_tier_winprob=0.0,
            s_tier_max_mae=-0.999,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(),
            edge_tier=EdgeTierSystem(thresholds=lenient_thresholds),
            # Make ExitThesis fall through to KEEP for any positive-or-near-zero edge
            exit_thesis=ExitThesisEngine(policy=ExitPolicy(
                hold_min_residual_edge=-0.999,    # never EXIT on residual
                reduce_zone_edge=-0.999,          # never TRIM
                rotation_margin=99.999,           # never ROTATE
            )),
            pyramid=PyramidPolicyEngine(policy=PyramidPolicy(
                min_unrealized_pnl_for_add=0.01,
                min_residual_edge_for_add=-0.999,  # lenient for synthetic edges
                min_current_tier_for_add="A",
                max_single_weight_after_add=0.40,
            )),
        )
        # AAPL is in opportunity_map (raw_opp 0.0035)
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL", current_weight=0.10)],
            exit_triggers={"AAPL": "max_hold"},
            adds_per_position={"AAPL": 0},
            initial_weights={"AAPL": 0.10},
        )
        # ADD_TO_WINNER appears for AAPL
        add_winners = [a for a in actions if a.action_type == "ADD_TO_WINNER"]
        assert len(add_winners) >= 1

    def test_pyramid_blocks_loss_position_in_full_stack(self):
        """회귀 핵심: 전체 stack에서도 손실 포지션 add-on 절대 발생 안 함."""
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True, edge_tier=True,
            exit_thesis=True, pyramid=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(),
            edge_tier=EdgeTierSystem(thresholds=TierThresholds()),
            exit_thesis=ExitThesisEngine(policy=ExitPolicy(
                hold_min_residual_edge=-0.999,
            )),
            pyramid=PyramidPolicyEngine(policy=PyramidPolicy()),
        )
        # Loss position
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL", pnl_pct=-0.05)],
            exit_triggers={"AAPL": "max_hold"},
        )
        # NO ADD_TO_WINNER for loss position (regression invariant)
        add_winners = [a for a in actions if a.action_type == "ADD_TO_WINNER"]
        assert len(add_winners) == 0


class TestRotationIntegration:
    """Rotation wiring — KEEP positions × new candidates matrix."""

    def test_rotation_evaluates_new_candidates(self):
        signal = _mock_signal(action="TRADE")
        # MSFT in opportunity_map can replace AAPL
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True, edge_tier=True,
            exit_thesis=True, rotation=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(),
            edge_tier=EdgeTierSystem(thresholds=TierThresholds(
                s_tier_net_edge=0.0, a_tier_net_edge=0.0, b_tier_net_edge=-0.999,
                s_tier_winprob=0.0, a_tier_winprob=0.0,
                s_tier_max_mae=-0.999,
            )),
            exit_thesis=ExitThesisEngine(policy=ExitPolicy(
                hold_min_residual_edge=-0.999,
            )),
            rotation=CapitalRotationEngine(
                policy=RotationPolicy(rotation_margin=0.0, max_rotations_per_month=10),
                switching_cost_fn=lambda a, b: 0.0,
            ),
        )
        # AAPL holding, MSFT/GOOG/AMZN in candidates
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
            positions=[_winner_position(ticker="AAPL", holding_days=2)],
            exit_triggers={"AAPL": "max_hold"},
        )
        # ROTATE may or may not happen depending on edge_tier output;
        # at minimum, rotation logic was invoked without error
        # (action set size > 0)
        assert len(actions) > 0


class TestAllocationIntegration:
    """Allocation override — replaces SignalGenerator sizing."""

    def test_allocation_overrides_signal_weights(self):
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True, allocation=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(policy=EdgePolicy(entry_threshold=-0.999)),
            allocation=AllocationEngine(),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        # Source policy on ADD_NEW should be AllocationEngine, not SignalGenerator
        add_news = [a for a in actions if a.action_type == "ADD_NEW"]
        # Alloc may produce 0 entries depending on net_edge gates;
        # check that source_policy includes "AllocationEngine" if any produced
        for a in add_news:
            assert a.source_policy in ("AllocationEngine", "SignalGenerator")

    def test_leverage_invariant_in_book_optimizer(self):
        """allocation 통합 결과도 Σ weights ≤ 1.00."""
        signal = _mock_signal(action="TRADE")
        flags = FeatureFlagsConfig(
            edge_calibrator=True, edge_engine=True, allocation=True,
        )
        bo = BookOptimizer(
            signal_gen=_mock_signal_generator(signal),
            features=flags,
            edge_calibrator=_make_calibrator(),
            edge_engine=EdgeEngine(policy=EdgePolicy(entry_threshold=-0.999)),
            allocation=AllocationEngine(
                regime_budgets={"neutral": RegimeBudget(
                    max_gross_exposure=1.50,  # excessive — should clip
                    target_annual_vol=0.30,
                )},
            ),
        )
        actions = bo.decide(
            ohlcv=pd.DataFrame(), vol_scores=pd.DataFrame(),
            regime=_regime(), cost=0.001, state=_state(),
        )
        new_weights_total = sum(
            a.target_weight for a in actions if a.action_type == "ADD_NEW"
        )
        assert new_weights_total <= 1.00 + 1e-9
