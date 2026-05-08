"""V3.3 ExitThesisEngine tests (PR-3.2).

Verifies:
  - DecisionPolicy protocol satisfied
  - decide() decision tree:
      · risk_exit → EXIT (never veto)
      · stale signal → refresh callback
      · no fresh signal → EXIT (conservative)
      · residual ≤ hold_min → EXIT
      · better replacement → ROTATE
      · residual < reduce_zone → TRIM
      · else → KEEP
  - _best_replacement (empty, all-low, multiple)
  - rotation hurdle (switching_cost + rotation_margin)
  - immutability
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.exit_thesis import (
    ExitPolicy,
    ExitThesisEngine,
)
from v3.strategy.types import EdgeCandidate, PositionState


def _position(
    ticker: str = "AAPL",
    current_weight: float = 0.30,
    holding_days: int = 5,
) -> PositionState:
    return PositionState(
        ticker=ticker,
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=210.0,
        current_weight=current_weight,
        entry_edge=0.008,
        residual_edge=0.005,
        entry_tier="A",
        current_tier="A",
        pnl_pct=0.05,
        max_unrealized_pnl_pct=0.06,
        drawdown_from_peak_pct=-0.01,
        holding_days=holding_days,
        dominant_alpha="trend",
        expected_holding_days=10,
        thesis_alive=True,
    )


def _candidate(
    ticker: str,
    net_edge: float | None = 0.005,
    raw_opp: float = 0.005,
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker=ticker,
        direction=0.05,
        conviction=0.7,
        raw_opportunity=raw_opp,
        regime="neutral",
        regime_score=0.5,
        net_edge_5d=net_edge,
        expected_return_5d=0.010,
        expected_mae_5d=-0.020,
        win_probability_5d=0.55,
    )


# ──────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────
class TestProtocolConformance:
    def test_implements_decision_policy(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        assert isinstance(eng, DecisionPolicy)

    def test_name(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        assert eng.name == "exit_thesis"


# ──────────────────────────────────────────────────────────────
# Risk exit never vetoed
# ──────────────────────────────────────────────────────────────
class TestRiskExitsNeverVetoed:
    @pytest.mark.parametrize("trigger", [
        "portfolio_stop", "vol_contraction",
        "dynamic_stop_mae", "circuit_breaker",
    ])
    def test_risk_exit_returns_exit_action(self, trigger):
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger=trigger,
            current_edge=_candidate("AAPL", net_edge=0.999),  # very positive
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"
        assert "risk_exit" in action.reason
        assert trigger in action.reason


# ──────────────────────────────────────────────────────────────
# Time-based decision tree
# ──────────────────────────────────────────────────────────────
class TestDecideTimeBasedExit:
    def test_residual_below_hold_min_exits(self):
        eng = ExitThesisEngine(policy=ExitPolicy(hold_min_residual_edge=0.0))
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=-0.001),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"
        assert "residual_edge_below_hold_min" in action.reason

    def test_residual_strong_returns_keep(self):
        eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
            reduce_zone_edge=0.0015,
        ))
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"
        assert "thesis_alive" in action.reason

    def test_residual_weak_returns_trim(self):
        eng = ExitThesisEngine(
            policy=ExitPolicy(
                hold_min_residual_edge=0.0,
                reduce_zone_edge=0.0030,
            ),
            trim_fraction=0.30,
        )
        # residual 0.001 < reduce_zone 0.0030 → TRIM 30%
        action = eng.decide(
            position=_position(current_weight=0.30),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.001),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "TRIM"
        # 30% trim → 0.30 × 0.70 = 0.21
        assert action.target_weight == pytest.approx(0.21, rel=1e-3)
        assert "edge_weakened" in action.reason


# ──────────────────────────────────────────────────────────────
# Replacement / Rotation
# ──────────────────────────────────────────────────────────────
class TestRotation:
    def test_better_replacement_triggers_rotation(self):
        eng = ExitThesisEngine(
            policy=ExitPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0010,
        )
        # current residual 0.005, replacement net_edge 0.020
        # improvement = 0.015, hurdle = switching 0.001 + margin 0.0025 = 0.0035
        # 0.015 > 0.0035 → ROTATE
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[_candidate("MSFT", net_edge=0.020)],
            signal_age_hours=0.0,
        )
        assert action.action_type == "ROTATE"
        assert action.replacement_ticker == "MSFT"

    def test_replacement_below_hurdle_keeps(self):
        eng = ExitThesisEngine(
            policy=ExitPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0010,
        )
        # current 0.005, replacement 0.0070 → improvement 0.0020 < hurdle 0.0035
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[_candidate("MSFT", net_edge=0.0070)],
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"

    def test_no_replacement_keeps(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[],  # no candidates
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"

    def test_replacement_lower_than_current_skipped(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.010),
            replacement_candidates=[
                _candidate("MSFT", net_edge=0.003),  # lower
                _candidate("GOOG", net_edge=0.005),  # also lower
            ],
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"


# ──────────────────────────────────────────────────────────────
# Stale signal refresh
# ──────────────────────────────────────────────────────────────
class TestStaleSignalRefresh:
    def test_stale_signal_calls_refresh_fn(self):
        refresh_calls = []

        def refresh(ticker: str) -> EdgeCandidate:
            refresh_calls.append(ticker)
            return _candidate(ticker, net_edge=0.008)

        eng = ExitThesisEngine(
            policy=ExitPolicy(max_signal_staleness_hours=8.0),
            signal_refresh_fn=refresh,
        )
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=-0.005),  # stale negative
            replacement_candidates=[],
            signal_age_hours=12.0,  # > 8h → refresh
        )
        assert refresh_calls == ["AAPL"]
        # After refresh: 0.008 > 0 → KEEP (not exit)
        assert action.action_type == "KEEP"

    def test_no_refresh_callback_uses_stale(self):
        eng = ExitThesisEngine(
            policy=ExitPolicy(max_signal_staleness_hours=8.0),
            signal_refresh_fn=None,
        )
        # Stale signal but no refresh callback → use stale
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[],
            signal_age_hours=12.0,
        )
        assert action.action_type == "KEEP"  # uses stale net_edge=0.005

    def test_refresh_returning_none_falls_through(self):
        eng = ExitThesisEngine(
            policy=ExitPolicy(max_signal_staleness_hours=8.0),
            signal_refresh_fn=lambda t: None,  # no fresh signal
        )
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=None,
            replacement_candidates=[],
            signal_age_hours=12.0,
        )
        # Refresh returned None and no current_edge → conservative EXIT
        assert action.action_type == "EXIT"
        assert "no_fresh_signal" in action.reason


# ──────────────────────────────────────────────────────────────
# No fresh signal — conservative exit
# ──────────────────────────────────────────────────────────────
class TestNoFreshSignal:
    def test_none_current_edge_exits(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=None,
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"
        assert "no_fresh_signal" in action.reason

    def test_none_net_edge_exits(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        c = EdgeCandidate(
            date=pd.Timestamp("2026-05-09"),
            ticker="AAPL", direction=0.05, conviction=0.7,
            raw_opportunity=0.005, regime="neutral", regime_score=0.5,
            # net_edge_5d = None
        )
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=c,
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"


# ──────────────────────────────────────────────────────────────
# _best_replacement helper
# ──────────────────────────────────────────────────────────────
class TestBestReplacement:
    def test_empty_returns_none(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        assert eng._best_replacement([], current_residual=0.005) is None

    def test_all_below_current_returns_none(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        cands = [
            _candidate("A", net_edge=0.001),
            _candidate("B", net_edge=0.003),
        ]
        assert eng._best_replacement(cands, current_residual=0.005) is None

    def test_picks_highest_net_edge(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        cands = [
            _candidate("A", net_edge=0.010),
            _candidate("B", net_edge=0.020),
            _candidate("C", net_edge=0.015),
        ]
        best = eng._best_replacement(cands, current_residual=0.005)
        assert best is not None
        assert best.ticker == "B"

    def test_filters_none_net_edge(self):
        eng = ExitThesisEngine(policy=ExitPolicy())
        cands = [
            _candidate("A", net_edge=None),
            _candidate("B", net_edge=0.020),
        ]
        best = eng._best_replacement(cands, current_residual=0.005)
        assert best is not None
        assert best.ticker == "B"


# ──────────────────────────────────────────────────────────────
# Immutability
# ──────────────────────────────────────────────────────────────
class TestImmutability:
    def test_book_action_frozen(self):
        from dataclasses import FrozenInstanceError
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            action.target_weight = 999.0  # type: ignore[misc]
