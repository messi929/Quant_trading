"""V3.3 SignalDecayEngine tests (PR-3.3).

Verifies:
  - DecisionPolicy protocol
  - expected_holding_days per profile + fallback
  - apply_decay (no reconfirm): exp 감소
  - apply_decay (reconfirm with S/A tier + same direction): 0.75 floor
  - reset 조건 false: low tier, opposite direction
  - dominant_alpha (positive max, negative max)
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.signal_decay import (
    AlphaProfile,
    SignalDecayEngine,
    SignalDecayPolicy,
)
from v3.strategy.types import EdgeCandidate, PositionState


def _position(
    dominant_alpha: str = "trend",
    holding_days: int = 5,
    entry_edge: float = 0.005,
) -> PositionState:
    return PositionState(
        ticker="AAPL",
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=210.0,
        current_weight=0.30,
        entry_edge=entry_edge,
        residual_edge=0.005,
        entry_tier="A",
        current_tier="A",
        pnl_pct=0.05,
        max_unrealized_pnl_pct=0.06,
        drawdown_from_peak_pct=-0.01,
        holding_days=holding_days,
        dominant_alpha=dominant_alpha,
        expected_holding_days=10,
        thesis_alive=True,
    )


def _signal(
    direction: float = 0.05,
    tier: str = "A",
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker="AAPL",
        direction=direction,
        conviction=0.7,
        raw_opportunity=0.005,
        regime="neutral",
        regime_score=0.5,
        net_edge_5d=0.005,
        edge_tier=tier,
    )


# ──────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────
class TestProtocolConformance:
    def test_implements_decision_policy(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        assert isinstance(eng, DecisionPolicy)

    def test_name(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        assert eng.name == "signal_decay"


# ──────────────────────────────────────────────────────────────
# AlphaProfile defaults
# ──────────────────────────────────────────────────────────────
class TestAlphaProfile:
    def test_trend_profile_long_horizon(self):
        policy = SignalDecayPolicy()
        trend = policy.alpha_profiles["trend"]
        assert trend.expected_holding_days == 10
        assert trend.decay_half_life_days == 5.0
        assert trend.allow_hold_extension is True

    def test_reversion_profile_short_horizon(self):
        policy = SignalDecayPolicy()
        rev = policy.alpha_profiles["reversion"]
        assert rev.expected_holding_days == 3
        assert rev.decay_half_life_days == 1.5
        assert rev.profit_take_fast is True


# ──────────────────────────────────────────────────────────────
# expected_holding_days / half_life
# ──────────────────────────────────────────────────────────────
class TestHoldingDays:
    def test_trend_long_horizon(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        assert eng.expected_holding_days("trend") == 10

    def test_reversion_short_horizon(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        assert eng.expected_holding_days("reversion") == 3

    def test_trend_longer_than_reversion(self):
        """Critical: alpha-specific differentiation works."""
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        assert eng.expected_holding_days("trend") > eng.expected_holding_days("reversion")

    def test_unknown_alpha_uses_default(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy(default_max_hold_days=5))
        assert eng.expected_holding_days("unknown_xyz") == 5


# ──────────────────────────────────────────────────────────────
# apply_decay — no reconfirm
# ──────────────────────────────────────────────────────────────
class TestApplyDecayNoReset:
    def test_zero_holding_no_decay(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        pos = _position(dominant_alpha="trend", holding_days=0)
        # exp(0) = 1.0
        result = eng.apply_decay(pos, current_value=0.010)
        assert result == pytest.approx(0.010)

    def test_decay_at_half_life(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        # trend half_life 5.0
        pos = _position(dominant_alpha="trend", holding_days=5)
        # exp(-5/5) = 1/e ≈ 0.368
        result = eng.apply_decay(pos, current_value=0.010)
        assert result == pytest.approx(0.010 * math.exp(-1.0), rel=1e-3)

    def test_reversion_decays_faster_than_trend(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        trend_pos = _position(dominant_alpha="trend", holding_days=3)
        rev_pos = _position(dominant_alpha="reversion", holding_days=3)
        trend_val = eng.apply_decay(trend_pos, current_value=0.010)
        rev_val = eng.apply_decay(rev_pos, current_value=0.010)
        # Reversion has shorter half-life → more decay → smaller value
        assert rev_val < trend_val


# ──────────────────────────────────────────────────────────────
# apply_decay — reconfirm reset
# ──────────────────────────────────────────────────────────────
class TestApplyDecayWithReset:
    def test_reconfirmed_floor_075(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        # Long holding → strong decay (exp(-100/5) ≈ 0)
        pos = _position(dominant_alpha="trend", holding_days=100,
                        entry_edge=0.010)
        signal = _signal(direction=0.05, tier="A")  # same sign positive
        result = eng.apply_decay(pos, current_value=0.010, current_signal=signal)
        # Reset floor 0.75 → 0.010 × 0.75 = 0.0075
        assert result == pytest.approx(0.0075)

    def test_low_tier_no_reset(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy(reset_min_tier="A"))
        pos = _position(dominant_alpha="trend", holding_days=100,
                        entry_edge=0.010)
        signal = _signal(direction=0.05, tier="B")  # B < A → no reset
        result = eng.apply_decay(pos, current_value=0.010, current_signal=signal)
        # Pure exp decay without floor → very small
        assert result < 0.001

    def test_opposite_direction_no_reset(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy(
            require_direction_same_sign_for_reset=True,
        ))
        pos = _position(dominant_alpha="trend", holding_days=100,
                        entry_edge=0.010)  # positive entry
        signal = _signal(direction=-0.05, tier="A")  # negative current
        result = eng.apply_decay(pos, current_value=0.010, current_signal=signal)
        # Direction flipped → no reset → pure exp decay
        assert result < 0.001

    def test_reconfirm_does_not_amplify_above_one(self):
        """Reset floor shouldn't make decay > 1 (no amplification)."""
        eng = SignalDecayEngine(policy=SignalDecayPolicy(reset_floor=0.75))
        pos = _position(dominant_alpha="trend", holding_days=1)
        # exp(-1/5) ≈ 0.819 > 0.75 → keep 0.819, no floor needed
        signal = _signal(direction=0.05, tier="A")
        result = eng.apply_decay(pos, current_value=0.010, current_signal=signal)
        # max(exp_decay, floor) = max(0.819, 0.75) = 0.819
        assert result == pytest.approx(0.010 * math.exp(-0.2), rel=1e-3)


# ──────────────────────────────────────────────────────────────
# is_reconfirmed
# ──────────────────────────────────────────────────────────────
class TestIsReconfirmed:
    def test_s_tier_same_sign_reconfirms(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        pos = _position(entry_edge=0.005)
        signal = _signal(direction=0.03, tier="S")
        assert eng.is_reconfirmed(pos, signal) is True

    def test_a_tier_same_sign_reconfirms(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy(reset_min_tier="A"))
        pos = _position(entry_edge=0.005)
        signal = _signal(direction=0.03, tier="A")
        assert eng.is_reconfirmed(pos, signal) is True

    def test_b_tier_no_reconfirm(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy(reset_min_tier="A"))
        pos = _position(entry_edge=0.005)
        signal = _signal(direction=0.03, tier="B")
        assert eng.is_reconfirmed(pos, signal) is False

    def test_none_tier_no_reconfirm(self):
        eng = SignalDecayEngine(policy=SignalDecayPolicy())
        pos = _position(entry_edge=0.005)
        signal = EdgeCandidate(
            date=pd.Timestamp("2026-05-09"), ticker="AAPL",
            direction=0.03, conviction=0.7, raw_opportunity=0.005,
            regime="neutral", regime_score=0.5,
            edge_tier=None,  # No tier
        )
        assert eng.is_reconfirmed(pos, signal) is False


# ──────────────────────────────────────────────────────────────
# dominant_alpha (static helper)
# ──────────────────────────────────────────────────────────────
class TestDominantAlpha:
    def test_positive_max(self):
        s = pd.Series({"trend": 0.05, "reversion": 0.02})
        assert SignalDecayEngine.dominant_alpha(s) == "trend"

    def test_negative_max(self):
        # |reversion| larger
        s = pd.Series({"trend": 0.02, "reversion": -0.08})
        assert SignalDecayEngine.dominant_alpha(s) == "reversion"

    def test_empty_series(self):
        assert SignalDecayEngine.dominant_alpha(pd.Series(dtype=float)) == ""

    def test_none_input(self):
        assert SignalDecayEngine.dominant_alpha(None) == ""
