"""V3.3 CapitalRotationEngine tests (PR-4.3).

Verifies:
  - DecisionPolicy protocol
  - rotation_edge > hurdle → ROTATE
  - hurdle 미달 → None (KEEP)
  - max_rotations_per_month 도달 → None
  - min_holding_days 미달 → None
  - candidate tier < min → None
  - lower tier replacing higher → None
  - disabled → None
  - switching_cost_fn 사용
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.rotation import (
    CapitalRotationEngine,
    RotationPolicy,
)
from v3.strategy.types import EdgeCandidate, PositionState


def _position(
    ticker: str = "AAPL",
    holding_days: int = 5,
    residual_edge: float = 0.005,
    current_tier: str = "A",
    current_weight: float = 0.20,
) -> PositionState:
    return PositionState(
        ticker=ticker,
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=210.0,
        current_weight=current_weight,
        entry_edge=0.005,
        residual_edge=residual_edge,
        entry_tier=current_tier,
        current_tier=current_tier,
        pnl_pct=0.05,
        max_unrealized_pnl_pct=0.06,
        drawdown_from_peak_pct=-0.005,
        holding_days=holding_days,
        dominant_alpha="trend",
        expected_holding_days=10,
        thesis_alive=True,
    )


def _candidate(
    ticker: str = "MSFT",
    net_edge: float = 0.020,
    tier: str = "A",
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker=ticker,
        direction=0.05,
        conviction=0.7,
        raw_opportunity=0.005,
        regime="neutral",
        regime_score=0.5,
        net_edge_5d=net_edge,
        edge_tier=tier,
        expected_return_5d=0.012,
        expected_mae_5d=-0.020,
        win_probability_5d=0.55,
    )


# ──────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────
class TestProtocol:
    def test_decision_policy(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        assert isinstance(eng, DecisionPolicy)

    def test_name(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        assert eng.name == "rotation"


# ──────────────────────────────────────────────────────────────
# Successful rotation
# ──────────────────────────────────────────────────────────────
class TestSuccessfulRotation:
    def test_high_edge_gap_triggers_rotation(self):
        eng = CapitalRotationEngine(
            policy=RotationPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0010,
        )
        # current residual 0.005, candidate 0.020 → gap 0.015
        # hurdle 0.0010 + 0.0025 = 0.0035 → 0.015 > 0.0035 → ROTATE
        action = eng.evaluate(
            position=_position(residual_edge=0.005),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is not None
        assert action.action_type == "ROTATE"
        assert action.replacement_ticker == "MSFT"
        assert action.target_weight == 0.0

    def test_action_metadata(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        action = eng.evaluate(
            position=_position(),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is not None
        assert "rotation_edge" in action.reason
        assert "hurdle" in action.reason
        assert action.source_policy == "rotation"


# ──────────────────────────────────────────────────────────────
# Hurdle 미달 → None
# ──────────────────────────────────────────────────────────────
class TestHurdleGate:
    def test_below_hurdle_returns_none(self):
        eng = CapitalRotationEngine(
            policy=RotationPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0010,
        )
        # gap 0.002 < hurdle 0.0035 → None
        action = eng.evaluate(
            position=_position(residual_edge=0.005),
            candidate=_candidate(net_edge=0.007),
            rotations_this_month=0,
        )
        assert action is None

    def test_exact_hurdle_returns_none(self):
        eng = CapitalRotationEngine(
            policy=RotationPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0010,
        )
        # gap exactly = hurdle (0.0035) → strictly greater required → None
        action = eng.evaluate(
            position=_position(residual_edge=0.005),
            candidate=_candidate(net_edge=0.0085),  # 0.005 + 0.0035
            rotations_this_month=0,
        )
        assert action is None

    def test_equal_residual_no_rotation(self):
        """Same edge → no rotation."""
        eng = CapitalRotationEngine(policy=RotationPolicy())
        action = eng.evaluate(
            position=_position(residual_edge=0.010),
            candidate=_candidate(net_edge=0.010),
            rotations_this_month=0,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# Monthly cap
# ──────────────────────────────────────────────────────────────
class TestMonthlyCap:
    def test_below_cap_allowed(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(max_rotations_per_month=4))
        action = eng.evaluate(
            position=_position(),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=3,
        )
        assert action is not None

    def test_at_cap_blocked(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(max_rotations_per_month=4))
        action = eng.evaluate(
            position=_position(),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=4,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# Min holding days
# ──────────────────────────────────────────────────────────────
class TestMinHoldingDays:
    def test_below_min_days_blocked(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(
            min_holding_days_before_rotation=2,
        ))
        action = eng.evaluate(
            position=_position(holding_days=1),  # < 2
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is None

    def test_at_min_days_allowed(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(
            min_holding_days_before_rotation=2,
        ))
        action = eng.evaluate(
            position=_position(holding_days=2),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is not None


# ──────────────────────────────────────────────────────────────
# Tier requirements
# ──────────────────────────────────────────────────────────────
class TestTierRequirements:
    def test_b_tier_candidate_blocked(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(
            min_candidate_tier="A",
        ))
        action = eng.evaluate(
            position=_position(current_tier="A"),
            candidate=_candidate(net_edge=0.020, tier="B"),
            rotations_this_month=0,
        )
        assert action is None

    def test_lower_tier_replacing_higher_blocked(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(
            forbid_lower_tier_replace_higher=True,
            min_candidate_tier="A",
        ))
        # Position S-tier, candidate A-tier → lower replacing higher
        action = eng.evaluate(
            position=_position(current_tier="S"),
            candidate=_candidate(net_edge=0.020, tier="A"),
            rotations_this_month=0,
        )
        assert action is None

    def test_s_replaces_a(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        action = eng.evaluate(
            position=_position(current_tier="A"),
            candidate=_candidate(net_edge=0.020, tier="S"),
            rotations_this_month=0,
        )
        assert action is not None

    def test_same_tier_allowed(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        action = eng.evaluate(
            position=_position(current_tier="A"),
            candidate=_candidate(net_edge=0.020, tier="A"),
            rotations_this_month=0,
        )
        assert action is not None


# ──────────────────────────────────────────────────────────────
# None net_edge sanity
# ──────────────────────────────────────────────────────────────
class TestNoneEdge:
    def test_none_candidate_edge_blocked(self):
        eng = CapitalRotationEngine(policy=RotationPolicy())
        action = eng.evaluate(
            position=_position(),
            candidate=_candidate(net_edge=None),
            rotations_this_month=0,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# switching_cost_fn 사용
# ──────────────────────────────────────────────────────────────
class TestSwitchingCost:
    def test_high_switching_cost_blocks(self):
        # gap 0.015, switching 0.020, margin 0.0025 → hurdle 0.0225
        # 0.015 < 0.0225 → None
        eng = CapitalRotationEngine(
            policy=RotationPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.020,
        )
        action = eng.evaluate(
            position=_position(residual_edge=0.005),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is None

    def test_zero_switching_cost(self):
        eng = CapitalRotationEngine(
            policy=RotationPolicy(rotation_margin=0.0025),
            switching_cost_fn=lambda a, b: 0.0,
        )
        # gap 0.015 > hurdle 0.0025 → ROTATE
        action = eng.evaluate(
            position=_position(residual_edge=0.005),
            candidate=_candidate(net_edge=0.020),
            rotations_this_month=0,
        )
        assert action is not None


# ──────────────────────────────────────────────────────────────
# Disabled
# ──────────────────────────────────────────────────────────────
class TestDisabled:
    def test_disabled_returns_none(self):
        eng = CapitalRotationEngine(policy=RotationPolicy(enabled=False))
        action = eng.evaluate(
            position=_position(),
            candidate=_candidate(net_edge=0.999),
            rotations_this_month=0,
        )
        assert action is None
