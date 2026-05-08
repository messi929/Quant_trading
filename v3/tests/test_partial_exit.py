"""V3.3 PartialExitEngine tests (PR-3.4).

Verifies:
  - vol_contraction → full EXIT
  - residual ≤ 0 → full EXIT
  - profit_protection (max +3% 후 -1.5% 하락 + edge 약화) → TRIM 50%
  - edge_weakening → TRIM 30%
  - trim 후 floor 미만 → EXIT 전환
  - 정상 KEEP
  - DecisionPolicy protocol
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.partial_exit import (
    PartialExitEngine,
    PartialExitPolicy,
)
from v3.strategy.types import PositionState


def _position(
    current_weight: float = 0.30,
    max_unrealized_pnl_pct: float = 0.05,
    drawdown_from_peak_pct: float = -0.005,
) -> PositionState:
    return PositionState(
        ticker="AAPL",
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=210.0,
        current_weight=current_weight,
        entry_edge=0.005,
        residual_edge=0.001,
        entry_tier="A",
        current_tier="A",
        pnl_pct=0.05,
        max_unrealized_pnl_pct=max_unrealized_pnl_pct,
        drawdown_from_peak_pct=drawdown_from_peak_pct,
        holding_days=5,
        dominant_alpha="trend",
        expected_holding_days=10,
        thesis_alive=True,
    )


# ──────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────
class TestProtocol:
    def test_decision_policy(self):
        eng = PartialExitEngine(policy=PartialExitPolicy())
        assert isinstance(eng, DecisionPolicy)

    def test_name(self):
        eng = PartialExitEngine(policy=PartialExitPolicy())
        assert eng.name == "partial_exit"


# ──────────────────────────────────────────────────────────────
# vol_contraction — always full EXIT
# ──────────────────────────────────────────────────────────────
class TestVolContraction:
    def test_vol_contraction_with_high_residual_still_exits(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            full_exit_on_vol_contraction=True
        ))
        action = eng.evaluate(
            position=_position(),
            residual_edge=0.999,  # very positive
            thesis_break="vol_contraction",
        )
        assert action.action_type == "EXIT"
        assert "vol_contraction" in action.reason

    def test_vol_contraction_disabled_falls_through(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            full_exit_on_vol_contraction=False
        ))
        action = eng.evaluate(
            position=_position(),
            residual_edge=0.005,
            thesis_break="vol_contraction",
        )
        # Falls through to KEEP
        assert action.action_type == "KEEP"


# ──────────────────────────────────────────────────────────────
# Negative residual → EXIT
# ──────────────────────────────────────────────────────────────
class TestNegativeResidual:
    def test_zero_residual_exits(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            exit_if_residual_edge_below=0.0
        ))
        action = eng.evaluate(
            position=_position(),
            residual_edge=0.0,
        )
        assert action.action_type == "EXIT"
        assert "residual_edge_negative" in action.reason

    def test_negative_residual_exits(self):
        eng = PartialExitEngine(policy=PartialExitPolicy())
        action = eng.evaluate(
            position=_position(),
            residual_edge=-0.005,
        )
        assert action.action_type == "EXIT"


# ──────────────────────────────────────────────────────────────
# Profit protection (max +3% 후 -1.5% 하락 + edge 약화)
# ──────────────────────────────────────────────────────────────
class TestProfitProtection:
    def test_protection_triggers_trim(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            profit_protection_trigger=0.030,
            drawdown_from_peak_trigger=-0.015,
            trim_if_residual_edge_below=0.002,
            trim_fraction_default=0.50,
        ))
        action = eng.evaluate(
            position=_position(
                current_weight=0.40,
                max_unrealized_pnl_pct=0.04,    # > 3%
                drawdown_from_peak_pct=-0.020,  # < -1.5%
            ),
            residual_edge=0.001,  # < 0.002
        )
        assert action.action_type == "TRIM"
        assert action.target_weight == pytest.approx(0.20)  # 50% trim
        assert "winner_protection" in action.reason

    def test_protection_no_trigger_when_no_drawdown(self):
        eng = PartialExitEngine(policy=PartialExitPolicy())
        action = eng.evaluate(
            position=_position(
                max_unrealized_pnl_pct=0.04,
                drawdown_from_peak_pct=-0.005,  # only -0.5%, not -1.5%
            ),
            residual_edge=0.001,
        )
        # Falls to weak_edge → 30% trim, not winner_protection
        assert action.action_type == "TRIM"
        assert "weak_residual_edge" in action.reason


# ──────────────────────────────────────────────────────────────
# Edge weakening — 30% trim
# ──────────────────────────────────────────────────────────────
class TestEdgeWeakening:
    def test_edge_below_threshold_trims_30pct(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            trim_if_residual_edge_below=0.002,
            weak_edge_trim_fraction=0.30,
        ))
        action = eng.evaluate(
            position=_position(current_weight=0.30),
            residual_edge=0.001,
        )
        assert action.action_type == "TRIM"
        # 30% trim → 0.30 × 0.70 = 0.21
        assert action.target_weight == pytest.approx(0.21)
        assert "weak_residual_edge" in action.reason


# ──────────────────────────────────────────────────────────────
# Floor enforcement (페르소나 정합성 — Phase 25.2)
# ──────────────────────────────────────────────────────────────
class TestFloorEnforcement:
    def test_trim_below_floor_converts_to_exit(self):
        # current 0.18, 50% trim → 0.09 < 0.15 floor → EXIT
        eng = PartialExitEngine(policy=PartialExitPolicy(
            min_remaining_weight=0.15,
            trim_fraction_default=0.50,
            profit_protection_trigger=0.030,
            drawdown_from_peak_trigger=-0.015,
            trim_if_residual_edge_below=0.002,
        ))
        action = eng.evaluate(
            position=_position(
                current_weight=0.18,
                max_unrealized_pnl_pct=0.04,
                drawdown_from_peak_pct=-0.020,
            ),
            residual_edge=0.001,
        )
        # Would trim to 0.09 < 0.15 floor → converted to EXIT
        assert action.action_type == "EXIT"
        assert "below_floor" in action.reason

    def test_trim_above_floor_stays_trim(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            min_remaining_weight=0.15,
            weak_edge_trim_fraction=0.30,
        ))
        action = eng.evaluate(
            position=_position(current_weight=0.40),  # 30% trim → 0.28 > 0.15
            residual_edge=0.001,
        )
        assert action.action_type == "TRIM"
        assert action.target_weight == pytest.approx(0.28)


# ──────────────────────────────────────────────────────────────
# KEEP path
# ──────────────────────────────────────────────────────────────
class TestKeep:
    def test_strong_residual_keeps(self):
        eng = PartialExitEngine(policy=PartialExitPolicy(
            trim_if_residual_edge_below=0.002,
        ))
        action = eng.evaluate(
            position=_position(),
            residual_edge=0.005,
        )
        assert action.action_type == "KEEP"
        assert action.target_weight == action.current_weight
        assert "thesis_alive" in action.reason


# ──────────────────────────────────────────────────────────────
# Priority order
# ──────────────────────────────────────────────────────────────
class TestPriorityOrder:
    def test_vol_contraction_outranks_residual(self):
        """Even with residual+strong, vol_contraction wins."""
        eng = PartialExitEngine(policy=PartialExitPolicy())
        action = eng.evaluate(
            position=_position(),
            residual_edge=0.999,
            thesis_break="vol_contraction",
        )
        assert action.action_type == "EXIT"
        assert "vol_contraction" in action.reason

    def test_negative_residual_outranks_winner_protection(self):
        eng = PartialExitEngine(policy=PartialExitPolicy())
        action = eng.evaluate(
            position=_position(
                max_unrealized_pnl_pct=0.04,
                drawdown_from_peak_pct=-0.020,
            ),
            residual_edge=-0.005,
        )
        # Negative residual → full EXIT, not trim
        assert action.action_type == "EXIT"
        assert "residual_edge_negative" in action.reason
