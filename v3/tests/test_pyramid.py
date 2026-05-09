"""V3.3 PyramidPolicyEngine tests (PR-4.2).

Verifies:
  - DecisionPolicy protocol
  - 손실 포지션 add-on 절대 금지 (회귀 핵심 invariant)
  - add-on 후 max_single_weight_after_add 위반 0
  - tier S/A pass, B fail
  - max_adds_per_position 제한
  - pnl_pct 미달 → None
  - residual_edge 미달 → None
  - new_weight ≤ current → None (sanity)
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.pyramid import PyramidPolicy, PyramidPolicyEngine
from v3.strategy.types import EdgeCandidate, PositionState


def _winner_position(
    pnl_pct: float = 0.05,
    current_weight: float = 0.20,
) -> PositionState:
    return PositionState(
        ticker="AAPL",
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=200.0 * (1.0 + pnl_pct),
        current_weight=current_weight,
        entry_edge=0.005,
        residual_edge=0.005,
        entry_tier="A",
        current_tier="A",
        pnl_pct=pnl_pct,
        max_unrealized_pnl_pct=max(pnl_pct, 0.0),
        drawdown_from_peak_pct=0.0,
        holding_days=4,
        dominant_alpha="trend",
        expected_holding_days=10,
        thesis_alive=True,
    )


def _candidate(
    net_edge: float = 0.005,
    tier: str = "A",
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker="AAPL",
        direction=0.05,
        conviction=0.7,
        raw_opportunity=0.005,
        regime="neutral",
        regime_score=0.5,
        net_edge_5d=net_edge,
        edge_tier=tier,
        expected_return_5d=0.010,
        expected_mae_5d=-0.020,
        win_probability_5d=0.55,
    )


# ──────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────
class TestProtocol:
    def test_decision_policy(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        assert isinstance(eng, DecisionPolicy)

    def test_name(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        assert eng.name == "pyramid"


# ──────────────────────────────────────────────────────────────
# 손실 포지션 add-on 절대 금지 (회귀 핵심)
# ──────────────────────────────────────────────────────────────
class TestNoAveragingDown:
    """이 invariant는 어떤 경우에도 깨져선 안 됨."""

    def test_loss_position_blocks_add(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=-0.05),  # 손실
            candidate=_candidate(net_edge=0.020, tier="S"),  # 강한 신호
            adds_so_far=0,
            initial_weight=0.20,
        )
        assert action is None

    def test_zero_pnl_blocks_add(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.0),  # break-even
            candidate=_candidate(net_edge=0.020, tier="S"),
            adds_so_far=0,
            initial_weight=0.20,
        )
        assert action is None

    @pytest.mark.parametrize("pnl", [-0.10, -0.05, -0.01, 0.0])
    def test_non_winner_always_blocked(self, pnl):
        """모든 non-winner pnl에서 add-on 차단."""
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=pnl),
            candidate=_candidate(net_edge=0.999, tier="S"),
            adds_so_far=0,
            initial_weight=0.30,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# Add-on 정상 경로
# ──────────────────────────────────────────────────────────────
class TestSuccessfulAddOn:
    def test_winner_with_strong_signal_adds(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05, current_weight=0.20),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=0,
            initial_weight=0.20,
        )
        assert action is not None
        assert action.action_type == "ADD_TO_WINNER"
        # initial 0.20 × 0.50 fraction = 0.10 added
        assert action.target_weight == pytest.approx(0.30)

    def test_action_metadata(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=0,
            initial_weight=0.20,
        )
        assert action.ticker == "AAPL"
        assert action.source_policy == "pyramid"
        assert "winner_add_on" in action.reason
        assert action.expected_impact == pytest.approx(0.010)


# ──────────────────────────────────────────────────────────────
# max_single_weight_after_add 강제
# ──────────────────────────────────────────────────────────────
class TestMaxWeightCap:
    def test_add_capped_at_max_single(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(
            max_single_weight_after_add=0.40,
            add_size_fraction_of_initial=0.50,
        ))
        # current 0.35, initial 0.30, 0.50 fraction → 0.15 add → 0.50
        # but max 0.40 → cap at 0.40
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05, current_weight=0.35),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=0,
            initial_weight=0.30,
        )
        assert action is not None
        assert action.target_weight <= 0.40 + 1e-9
        assert action.target_weight == pytest.approx(0.40)

    def test_already_at_max_no_add(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(
            max_single_weight_after_add=0.40,
        ))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05, current_weight=0.40),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=0,
            initial_weight=0.30,
        )
        # new_weight == current → no actual add → None
        assert action is None


# ──────────────────────────────────────────────────────────────
# Tier requirement
# ──────────────────────────────────────────────────────────────
class TestTierRequirement:
    def test_s_tier_passes(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(min_current_tier_for_add="A"))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="S"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is not None

    def test_a_tier_passes(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(min_current_tier_for_add="A"))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is not None

    def test_b_tier_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(min_current_tier_for_add="A"))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="B"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None

    def test_blocked_tier_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(min_current_tier_for_add="A"))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="BLOCKED"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# max_adds_per_position 제한
# ──────────────────────────────────────────────────────────────
class TestMaxAdds:
    def test_below_max_allowed(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(max_adds_per_position=2))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=1, initial_weight=0.20,
        )
        assert action is not None

    def test_at_max_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(max_adds_per_position=2))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=2, initial_weight=0.20,
        )
        assert action is None

    def test_above_max_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(max_adds_per_position=2))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.010, tier="A"),
            adds_so_far=10, initial_weight=0.20,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# Threshold gates
# ──────────────────────────────────────────────────────────────
class TestThresholdGates:
    def test_pnl_below_min_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(
            min_unrealized_pnl_for_add=0.015,
        ))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.010),  # < 0.015
            candidate=_candidate(net_edge=0.020, tier="S"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None

    def test_residual_below_min_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(
            min_residual_edge_for_add=0.004,
        ))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.002, tier="A"),  # < 0.004
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None

    def test_none_residual_blocked(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=None, tier="A"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None


# ──────────────────────────────────────────────────────────────
# Disabled policy
# ──────────────────────────────────────────────────────────────
class TestDisabled:
    def test_disabled_returns_none(self):
        eng = PyramidPolicyEngine(policy=PyramidPolicy(enabled=False))
        action = eng.evaluate(
            position=_winner_position(pnl_pct=0.05),
            candidate=_candidate(net_edge=0.020, tier="S"),
            adds_so_far=0, initial_weight=0.20,
        )
        assert action is None
