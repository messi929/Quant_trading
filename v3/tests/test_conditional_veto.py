"""V3.3 Conditional Veto tests (PR-3.1, FOLLOW_UPS 1순위).

Verifies:
  - is_risk_based_exit / is_time_based_exit
  - should_refresh_signal (8h vs 16h threshold)
  - evaluate_conditional_veto:
      · risk_exit는 항상 should_exit=True (veto 금지)
      · time_based + residual_edge > 0 → KEEP (veto)
      · time_based + residual_edge ≤ 0 → exit
  - 4/21 ADI +9.02% 회귀 시나리오 (profit_take vetoed by positive residual_edge)
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy.exit_thesis import (
    ExitPolicy,
    RISK_BASED_EXITS,
    TIME_BASED_EXITS,
    evaluate_conditional_veto,
    is_risk_based_exit,
    is_time_based_exit,
    should_refresh_signal,
    signal_age_hours,
)


# ──────────────────────────────────────────────────────────────
# Trigger taxonomy
# ──────────────────────────────────────────────────────────────
class TestTriggerTaxonomy:
    def test_risk_based_set(self):
        # All 4 risk exits
        assert "portfolio_stop" in RISK_BASED_EXITS
        assert "dynamic_stop_mae" in RISK_BASED_EXITS
        assert "vol_contraction" in RISK_BASED_EXITS
        assert "circuit_breaker" in RISK_BASED_EXITS
        assert len(RISK_BASED_EXITS) == 4

    def test_time_based_set(self):
        assert "profit_take" in TIME_BASED_EXITS
        assert "max_hold" in TIME_BASED_EXITS
        assert len(TIME_BASED_EXITS) == 2

    def test_no_overlap(self):
        assert RISK_BASED_EXITS.isdisjoint(TIME_BASED_EXITS)

    def test_is_risk_based(self):
        assert is_risk_based_exit("portfolio_stop")
        assert is_risk_based_exit("vol_contraction")
        assert not is_risk_based_exit("profit_take")
        assert not is_risk_based_exit("max_hold")
        assert not is_risk_based_exit("unknown")

    def test_is_time_based(self):
        assert is_time_based_exit("profit_take")
        assert is_time_based_exit("max_hold")
        assert not is_time_based_exit("portfolio_stop")
        assert not is_time_based_exit("unknown")


# ──────────────────────────────────────────────────────────────
# evaluate_conditional_veto — RISK never vetoed
# ──────────────────────────────────────────────────────────────
class TestRiskExitNeverVetoed:
    """Critical invariant — risk_exit는 어떤 경우에도 청산되어야."""

    @pytest.mark.parametrize("trigger", list(RISK_BASED_EXITS))
    def test_risk_exit_with_high_positive_residual(self, trigger):
        policy = ExitPolicy()
        should_exit, reason = evaluate_conditional_veto(
            trigger, residual_edge=0.999, policy=policy,
        )
        assert should_exit is True
        assert "risk_exit" in reason
        assert trigger in reason

    @pytest.mark.parametrize("trigger", list(RISK_BASED_EXITS))
    def test_risk_exit_with_zero_residual(self, trigger):
        policy = ExitPolicy()
        should_exit, _ = evaluate_conditional_veto(
            trigger, residual_edge=0.0, policy=policy,
        )
        assert should_exit is True

    @pytest.mark.parametrize("trigger", list(RISK_BASED_EXITS))
    def test_risk_exit_with_negative_residual(self, trigger):
        policy = ExitPolicy()
        should_exit, _ = evaluate_conditional_veto(
            trigger, residual_edge=-0.05, policy=policy,
        )
        assert should_exit is True


# ──────────────────────────────────────────────────────────────
# evaluate_conditional_veto — TIME-based veto logic
# ──────────────────────────────────────────────────────────────
class TestTimeBasedConditionalVeto:
    def test_profit_take_with_positive_residual_kept(self):
        policy = ExitPolicy(hold_min_residual_edge=0.0)
        should_exit, reason = evaluate_conditional_veto(
            "profit_take", residual_edge=0.005, policy=policy,
        )
        assert should_exit is False  # KEEP (vetoed)
        assert "vetoed" in reason
        assert "profit_take" in reason

    def test_profit_take_with_negative_residual_exits(self):
        policy = ExitPolicy(hold_min_residual_edge=0.0)
        should_exit, reason = evaluate_conditional_veto(
            "profit_take", residual_edge=-0.001, policy=policy,
        )
        assert should_exit is True
        assert "residual_edge_below_hold_min" in reason

    def test_max_hold_with_positive_residual_kept(self):
        policy = ExitPolicy(hold_min_residual_edge=0.0)
        should_exit, _ = evaluate_conditional_veto(
            "max_hold", residual_edge=0.003, policy=policy,
        )
        assert should_exit is False

    def test_threshold_exact_boundary_exits(self):
        # residual_edge == hold_min → not strictly greater → exit
        policy = ExitPolicy(hold_min_residual_edge=0.0)
        should_exit, _ = evaluate_conditional_veto(
            "profit_take", residual_edge=0.0, policy=policy,
        )
        assert should_exit is True

    def test_custom_hold_min_threshold(self):
        # Higher hold_min means stricter veto
        policy = ExitPolicy(hold_min_residual_edge=0.005)
        # Residual 0.003 < 0.005 → exit
        should_exit, _ = evaluate_conditional_veto(
            "profit_take", residual_edge=0.003, policy=policy,
        )
        assert should_exit is True
        # Residual 0.010 > 0.005 → keep
        should_exit2, _ = evaluate_conditional_veto(
            "profit_take", residual_edge=0.010, policy=policy,
        )
        assert should_exit2 is False


# ──────────────────────────────────────────────────────────────
# evaluate_conditional_veto — Unknown triggers safe-exit
# ──────────────────────────────────────────────────────────────
class TestUnknownTrigger:
    def test_unknown_trigger_exits_by_default(self):
        policy = ExitPolicy()
        should_exit, reason = evaluate_conditional_veto(
            "some_unknown_trigger", residual_edge=0.999, policy=policy,
        )
        assert should_exit is True
        assert "unknown_trigger" in reason


# ──────────────────────────────────────────────────────────────
# Stale signal threshold
# ──────────────────────────────────────────────────────────────
class TestSignalRefresh:
    def test_v322_8h_threshold_was_too_short(self):
        """V3.2.1 bug: 8h < KR↔US 14h gap → always stale.
        V3.3 default 16h fixes this."""
        policy = ExitPolicy(max_signal_staleness_hours=16.0)
        # 14h (KR↔US gap) NOT stale under V3.3
        assert should_refresh_signal(14.0, policy) is False

    def test_above_threshold_stale(self):
        policy = ExitPolicy(max_signal_staleness_hours=16.0)
        assert should_refresh_signal(17.0, policy) is True

    def test_at_exact_threshold_not_stale(self):
        policy = ExitPolicy(max_signal_staleness_hours=16.0)
        # > comparison means equal is not stale
        assert should_refresh_signal(16.0, policy) is False

    def test_zero_age_fresh(self):
        policy = ExitPolicy()
        assert should_refresh_signal(0.0, policy) is False


# ──────────────────────────────────────────────────────────────
# signal_age_hours
# ──────────────────────────────────────────────────────────────
class TestSignalAge:
    def test_one_hour_diff(self):
        gen = pd.Timestamp("2026-05-09 09:00:00")
        now = pd.Timestamp("2026-05-09 10:00:00")
        assert signal_age_hours(gen, now) == pytest.approx(1.0)

    def test_kr_to_us_gap_14h(self):
        # KR session: 09:30 → US session: 23:30 (14h later)
        kr = pd.Timestamp("2026-05-09 09:30:00")
        us = pd.Timestamp("2026-05-09 23:30:00")
        assert signal_age_hours(kr, us) == pytest.approx(14.0)


# ──────────────────────────────────────────────────────────────
# 회귀 시나리오: 4/21 ADI +9.02%
# ──────────────────────────────────────────────────────────────
class TestADIRegression:
    """4/21 paper trade: ADI +9.02% TP fired but signal was stale.

    V3.2.1 buggy path: 8h staleness threshold + KR↔US 14h gap → stale
    → conditional veto skipped → unconditional exit at +9.02% (cut winner).

    V3.3 fix: 16h staleness threshold lets signal stay valid across sessions
    → residual_edge re-evaluation runs → if positive, KEEP (winner protected).
    """

    def test_adi_scenario_with_positive_residual_keeps_winner(self):
        # Simulate Day 4 of ADI hold:
        #   profit_take triggered (+9.02% > target)
        #   residual_edge from fresh signal: 0.005 (still positive — alpha alive)
        policy = ExitPolicy(
            hold_min_residual_edge=0.0,
            max_signal_staleness_hours=16.0,
        )
        # Signal generated 14h ago (KR session) — still fresh under V3.3
        assert should_refresh_signal(14.0, policy) is False

        should_exit, reason = evaluate_conditional_veto(
            "profit_take", residual_edge=0.005, policy=policy,
        )
        # WINNER PROTECTED — would have been cut in V3.2.1
        assert should_exit is False
        assert "vetoed" in reason

    def test_v322_8h_path_would_have_failed(self):
        """Document V3.2.1 bug for reference. With 8h threshold, 14h was
        considered stale → veto bypassed. We don't reproduce the bug,
        we verify V3.3 default 16h doesn't trigger it."""
        v322_buggy_policy = ExitPolicy(max_signal_staleness_hours=8.0)
        # KR↔US 14h gap → stale under buggy policy
        assert should_refresh_signal(14.0, v322_buggy_policy) is True
        # V3.3 default 16h does NOT mark 14h as stale
        v33_policy = ExitPolicy(max_signal_staleness_hours=16.0)
        assert should_refresh_signal(14.0, v33_policy) is False
