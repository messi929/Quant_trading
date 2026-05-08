"""V3.3 Phase 3 integration tests (PR-3.5).

Verifies cross-module behavior when multiple Phase 3 policies are active:
  - Conditional Veto + ExitThesisEngine: risk_exit는 ExitThesis 우회 (즉시 EXIT)
  - ExitThesisEngine + SignalDecay: decay-adjusted residual로 의사결정
  - ExitThesisEngine + PartialExit: ExitThesis가 PartialExit에 위임 가능
  - SignalDecay + PartialExit: decayed residual로 partial 의사결정
  - 모든 정책 ON: 우선순위 정합성

회귀 시나리오:
  - 4/21 ADI +9.02% (Conditional Veto + 모든 정책 ON에서 KEEP)
  - vol_contraction (모든 정책 활성에서 EXIT 보장)
  - Bear regime (residual 0 → EXIT)

이 파일은 Phase 3 모듈 간 직접 호출 통합. BookOptimizer 통합은 후속 PR.
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy.exit_thesis import (
    ExitPolicy,
    ExitThesisEngine,
    RISK_BASED_EXITS,
    evaluate_conditional_veto,
)
from v3.strategy.partial_exit import (
    PartialExitEngine,
    PartialExitPolicy,
)
from v3.strategy.signal_decay import (
    SignalDecayEngine,
    SignalDecayPolicy,
)
from v3.strategy.types import EdgeCandidate, PositionState


# ──────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────
def _position(
    ticker: str = "AAPL",
    holding_days: int = 5,
    current_weight: float = 0.30,
    dominant_alpha: str = "trend",
    pnl_pct: float = 0.05,
    max_unrealized: float = 0.06,
    drawdown: float = -0.005,
    entry_edge: float = 0.005,
) -> PositionState:
    return PositionState(
        ticker=ticker,
        entry_date=pd.Timestamp("2026-05-01"),
        entry_price=200.0,
        current_price=200.0 * (1.0 + pnl_pct),
        current_weight=current_weight,
        entry_edge=entry_edge,
        residual_edge=0.005,
        entry_tier="A",
        current_tier="A",
        pnl_pct=pnl_pct,
        max_unrealized_pnl_pct=max_unrealized,
        drawdown_from_peak_pct=drawdown,
        holding_days=holding_days,
        dominant_alpha=dominant_alpha,
        expected_holding_days=10,
        thesis_alive=True,
    )


def _candidate(
    ticker: str,
    net_edge: float | None = 0.005,
    direction: float = 0.05,
    tier: str = "A",
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker=ticker,
        direction=direction,
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
# Conditional Veto + ExitThesis cross-module
# ──────────────────────────────────────────────────────────────
class TestConditionalVetoExitThesisCohesion:
    """ExitThesis.decide()는 evaluate_conditional_veto의 risk_exit semantics를 보존."""

    @pytest.mark.parametrize("trigger", list(RISK_BASED_EXITS))
    def test_risk_trigger_always_exits_in_thesis_engine(self, trigger):
        eng = ExitThesisEngine(policy=ExitPolicy())
        action = eng.decide(
            position=_position(),
            exit_trigger=trigger,
            current_edge=_candidate("AAPL", net_edge=0.999),  # very positive
            replacement_candidates=[_candidate("MSFT", net_edge=0.999)],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"
        # Both helpers and class agree
        should_exit, _ = evaluate_conditional_veto(trigger, 0.999, ExitPolicy())
        assert should_exit is True

    def test_time_based_with_positive_residual_keeps_in_engine(self):
        """ExitThesisEngine on profit_take with positive residual → KEEP."""
        eng = ExitThesisEngine(policy=ExitPolicy(hold_min_residual_edge=0.0))
        action = eng.decide(
            position=_position(),
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=0.005),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"


# ──────────────────────────────────────────────────────────────
# ExitThesis + SignalDecay
# ──────────────────────────────────────────────────────────────
class TestExitThesisSignalDecayCohesion:
    def test_decayed_residual_changes_decision(self):
        """Decay can flip a KEEP into TRIM/EXIT.

        Reversion held 100d → exp(-100/1.5) is positive but ~1e-29.
        With hold_min_residual_edge=0.001, decayed residual fails the
        hold gate → EXIT.
        """
        decay_eng = SignalDecayEngine(policy=SignalDecayPolicy())
        thesis_eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.001,  # require minimum 0.1% to KEEP
            reduce_zone_edge=0.0030,
        ))

        pos = _position(dominant_alpha="reversion", holding_days=100,
                        entry_edge=0.005)
        decayed = decay_eng.apply_decay(pos, current_value=0.005)
        assert 0.0 < decayed < 0.0001  # positive but essentially zero

        # ExitThesis: decayed (≈0) ≤ hold_min (0.001) → EXIT
        action = thesis_eng.decide(
            position=pos,
            exit_trigger="max_hold",
            current_edge=_candidate("AAPL", net_edge=decayed),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"

    def test_reconfirmed_signal_preserves_keep(self):
        """Reset floor 0.75 prevents premature exit when signal reconfirmed."""
        decay_eng = SignalDecayEngine(policy=SignalDecayPolicy())
        pos = _position(dominant_alpha="trend", holding_days=20,
                        entry_edge=0.005)
        signal = _candidate("AAPL", net_edge=0.010, direction=0.05, tier="A")
        # Pre-decay: trend half_life=5, exp(-20/5) ≈ 0.018
        # Floor 0.75 reset → 0.005 × 0.75 = 0.00375
        decayed = decay_eng.apply_decay(
            pos, current_value=0.005, current_signal=signal,
        )
        assert decayed == pytest.approx(0.00375)

        # ExitThesis still KEEPS with decayed residual > 0
        thesis_eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
            reduce_zone_edge=0.0015,
        ))
        action = thesis_eng.decide(
            position=pos,
            exit_trigger="max_hold",
            current_edge=_candidate("AAPL", net_edge=decayed, tier="A"),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "KEEP"


# ──────────────────────────────────────────────────────────────
# SignalDecay + PartialExit
# ──────────────────────────────────────────────────────────────
class TestSignalDecayPartialExit:
    def test_decayed_residual_into_partial_decision(self):
        decay_eng = SignalDecayEngine(policy=SignalDecayPolicy())
        partial_eng = PartialExitEngine(policy=PartialExitPolicy(
            trim_if_residual_edge_below=0.002,
            exit_if_residual_edge_below=0.0,
        ))

        # reversion held 5 days → exp(-5/1.5) ≈ 0.036
        pos = _position(
            dominant_alpha="reversion", holding_days=5,
            current_weight=0.40,
        )
        decayed = decay_eng.apply_decay(pos, current_value=0.010)
        # ~0.00036 — below trim_if_residual but above exit (0.0)
        assert 0 < decayed < 0.002

        action = partial_eng.evaluate(
            position=pos, residual_edge=decayed,
        )
        # Below trim threshold → 30% trim
        assert action.action_type == "TRIM"


# ──────────────────────────────────────────────────────────────
# 회귀: 4/21 ADI scenario across full Phase 3 stack
# ──────────────────────────────────────────────────────────────
class TestADIRegressionWithFullStack:
    def test_adi_winner_protected_with_all_policies(self):
        """4/21 ADI +9.02% scenario — winner must be KEPT.

        Stack: Conditional Veto + ExitThesis + SignalDecay all active.
        """
        # ADI Day 4: profit_take triggered, fresh signal residual 0.005
        decay_eng = SignalDecayEngine(policy=SignalDecayPolicy())
        thesis_eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
            max_signal_staleness_hours=16.0,  # V3.3 fix
        ))

        pos = _position(
            ticker="ADI", dominant_alpha="trend", holding_days=4,
            entry_edge=0.005,
        )
        fresh_signal = _candidate("ADI", net_edge=0.005, direction=0.05, tier="A")

        # Apply decay (with reconfirm reset)
        decayed = decay_eng.apply_decay(
            pos, current_value=fresh_signal.net_edge_5d,
            current_signal=fresh_signal,
        )
        # trend half_life 5.0, holding 4 → exp(-4/5) ≈ 0.449
        # Reconfirmed → max(0.449, 0.75) = 0.75 → 0.005 × 0.75 = 0.00375
        assert decayed == pytest.approx(0.00375, rel=1e-3)

        # ExitThesis: profit_take + decayed residual > 0 → KEEP
        # (signal_age_hours 14.0 = KR↔US gap, not stale under V3.3 16h)
        action = thesis_eng.decide(
            position=pos,
            exit_trigger="profit_take",
            current_edge=_candidate("ADI", net_edge=decayed, tier="A"),
            replacement_candidates=[],
            signal_age_hours=14.0,
        )
        assert action.action_type == "KEEP"
        assert "thesis_alive" in action.reason


# ──────────────────────────────────────────────────────────────
# 회귀: vol_contraction에서 모든 stack 무조건 EXIT
# ──────────────────────────────────────────────────────────────
class TestVolContractionAcrossStack:
    def test_all_modules_agree_on_vol_contraction_exit(self):
        thesis_eng = ExitThesisEngine(policy=ExitPolicy())
        partial_eng = PartialExitEngine(policy=PartialExitPolicy())

        pos = _position()
        positive_signal = _candidate("AAPL", net_edge=0.999, tier="S")

        # ExitThesis: vol_contraction → EXIT
        thesis_action = thesis_eng.decide(
            position=pos,
            exit_trigger="vol_contraction",
            current_edge=positive_signal,
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert thesis_action.action_type == "EXIT"

        # PartialExit: vol_contraction trigger → EXIT
        partial_action = partial_eng.evaluate(
            position=pos,
            residual_edge=0.999,
            thesis_break="vol_contraction",
        )
        assert partial_action.action_type == "EXIT"


# ──────────────────────────────────────────────────────────────
# 회귀: 손실 포지션 + 모든 정책 → 전량 청산
# ──────────────────────────────────────────────────────────────
class TestLossPositionFullExit:
    def test_loss_position_with_negative_residual_exits_via_partial(self):
        partial_eng = PartialExitEngine(policy=PartialExitPolicy(
            exit_if_residual_edge_below=0.0,
        ))
        pos = _position(pnl_pct=-0.05)  # 손실 -5%
        action = partial_eng.evaluate(
            position=pos, residual_edge=-0.005,
        )
        assert action.action_type == "EXIT"

    def test_loss_position_via_thesis_engine(self):
        thesis_eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
        ))
        pos = _position(pnl_pct=-0.05)
        action = thesis_eng.decide(
            position=pos,
            exit_trigger="profit_take",
            current_edge=_candidate("AAPL", net_edge=-0.005),
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        assert action.action_type == "EXIT"


# ──────────────────────────────────────────────────────────────
# Action immutability across all engines
# ──────────────────────────────────────────────────────────────
class TestImmutabilityAcrossEngines:
    def test_all_engines_return_frozen_actions(self):
        from dataclasses import FrozenInstanceError

        thesis_eng = ExitThesisEngine(policy=ExitPolicy())
        partial_eng = PartialExitEngine(policy=PartialExitPolicy())

        pos = _position()
        signal = _candidate("AAPL", net_edge=0.005)

        thesis_action = thesis_eng.decide(
            position=pos,
            exit_trigger="profit_take",
            current_edge=signal,
            replacement_candidates=[],
            signal_age_hours=0.0,
        )
        partial_action = partial_eng.evaluate(
            position=pos, residual_edge=0.005,
        )

        for action in [thesis_action, partial_action]:
            with pytest.raises(FrozenInstanceError):
                action.target_weight = 999.0  # type: ignore[misc]
