"""V3.3 paper trade regression scenarios from 2026-04-11 ~ 05-03 (P8).

Encodes documented paper trades to verify V3.3 policy behavior produces
the expected (corrected) decision vs V3.2.1 actual outcome.

Known scenarios (from CLAUDE.md / FOLLOW_UPS.md / paper logs):
  ADI Day 4 +9.02% — Conditional Veto stale fire (V3.2.1 cut winner)
  AMZN Day 3 +3.30% — profit_take stale fire
  FANG 18% weight winner — pyramid candidate (V3.2.1 single trance)
  Loss position (negative PnL trade) — pyramid block invariant

Each scenario verifies V3.3 policy decision against documented context.
Server-side full 14-trade extraction is a future PR (requires
paper_account.json access).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from v3.strategy.exit_thesis import (
    ExitPolicy,
    ExitThesisEngine,
    evaluate_conditional_veto,
)
from v3.strategy.partial_exit import (
    PartialExitEngine,
    PartialExitPolicy,
)
from v3.strategy.pyramid import PyramidPolicy, PyramidPolicyEngine
from v3.strategy.types import EdgeCandidate, PositionState


# ──────────────────────────────────────────────────────────────
# Scenario fixture
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PaperTradeScenario:
    """Documented paper trade for regression encoding."""
    name: str
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str            # V3.2.1 actual exit reason
    pnl_pct: float              # realized PnL %
    holding_days: int
    raw_opportunity: float      # entry-time opportunity (from logs)
    weight_at_entry: float
    notes: str = ""

    def to_position_state(
        self,
        residual_edge: float,
        current_tier: str = "A",
    ) -> PositionState:
        """Build PositionState matching scenario at exit decision moment."""
        return PositionState(
            ticker=self.ticker,
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            current_price=self.exit_price,
            current_weight=self.weight_at_entry,
            entry_edge=self.raw_opportunity,
            residual_edge=residual_edge,
            entry_tier=current_tier,
            current_tier=current_tier,
            pnl_pct=self.pnl_pct,
            max_unrealized_pnl_pct=max(self.pnl_pct, 0.0),
            drawdown_from_peak_pct=-0.005,
            holding_days=self.holding_days,
            dominant_alpha="trend",
            expected_holding_days=10,
            thesis_alive=self.pnl_pct > 0,
        )


# ──────────────────────────────────────────────────────────────
# Documented scenarios
# ──────────────────────────────────────────────────────────────
SCENARIOS: dict[str, PaperTradeScenario] = {
    "adi_winner_cut": PaperTradeScenario(
        name="ADI_4_21_winner_cut",
        ticker="ADI",
        entry_date=pd.Timestamp("2026-04-17"),
        exit_date=pd.Timestamp("2026-04-21"),
        entry_price=200.0,
        exit_price=218.04,            # +9.02%
        exit_reason="profit_take",
        pnl_pct=0.0902,
        holding_days=4,
        raw_opportunity=0.0050,
        weight_at_entry=0.18,
        notes=(
            "V3.2.1 stale signal (KR↔US 14h > 8h threshold) → "
            "Conditional Veto bypassed → winner cut at +9.02%. "
            "V3.3 fix: 16h threshold + refresh_at_session_start → KEEP."
        ),
    ),
    "amzn_winner_cut": PaperTradeScenario(
        name="AMZN_winner_cut",
        ticker="AMZN",
        entry_date=pd.Timestamp("2026-04-19"),
        exit_date=pd.Timestamp("2026-04-22"),
        entry_price=180.0,
        exit_price=185.94,            # +3.30%
        exit_reason="profit_take",
        pnl_pct=0.0330,
        holding_days=3,
        raw_opportunity=0.0040,
        weight_at_entry=0.07,
        notes=(
            "V3.2.1 same stale-signal bug — winner cut at +3.30%. "
            "V3.3 expected: KEEP if residual edge > 0."
        ),
    ),
    "fang_undersized_winner": PaperTradeScenario(
        name="FANG_undersized_winner",
        ticker="FANG",
        entry_date=pd.Timestamp("2026-04-15"),
        exit_date=pd.Timestamp("2026-04-22"),
        entry_price=160.0,
        exit_price=164.16,            # +2.6%
        exit_reason="max_hold",
        pnl_pct=0.026,
        holding_days=5,
        raw_opportunity=0.0060,
        weight_at_entry=0.18,         # below max 0.40 → pyramid candidate
        notes=(
            "Largest weight in 4월 paper (18%, max 40% 미사용). "
            "V3.3 Pyramid: pnl > 1.5% AND tier ≥ A → ADD_TO_WINNER."
        ),
    ),
    "loss_position_blocked_add": PaperTradeScenario(
        name="loss_position_pyramid_blocked",
        ticker="LOSS_TICKER",
        entry_date=pd.Timestamp("2026-04-25"),
        exit_date=pd.Timestamp("2026-04-29"),
        entry_price=100.0,
        exit_price=98.0,              # -2.0%
        exit_reason="profit_take",    # never reached actually
        pnl_pct=-0.020,
        holding_days=4,
        raw_opportunity=0.0030,
        weight_at_entry=0.10,
        notes=(
            "Loss position invariant test. V3.3 Pyramid must NEVER add "
            "to averaging-down regardless of signal strength."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────
# ADI scenario — Conditional Veto fix
# ──────────────────────────────────────────────────────────────
class TestADIWinnerProtected:
    def test_v322_buggy_path_would_have_cut(self):
        """Document V3.2.1 bug: 8h threshold makes 14h KR↔US gap stale."""
        from v3.strategy.exit_thesis import should_refresh_signal
        v322_policy = ExitPolicy(max_signal_staleness_hours=8.0)
        # KR↔US gap = 14h → stale under V3.2.1 → unconditional fire
        assert should_refresh_signal(14.0, v322_policy)

    def test_v33_kept_with_positive_residual(self):
        """V3.3 fix: 16h threshold + positive residual_edge → KEEP."""
        scen = SCENARIOS["adi_winner_cut"]
        # Day 4, residual_edge still positive (alpha alive)
        residual = 0.005

        # Conditional Veto helper
        should_exit, reason = evaluate_conditional_veto(
            scen.exit_reason, residual,
            ExitPolicy(hold_min_residual_edge=0.0),
        )
        assert not should_exit, f"V3.3 should KEEP ADI: reason={reason}"

        # Full ExitThesisEngine
        eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
            reduce_zone_edge=0.001,
            max_signal_staleness_hours=16.0,
        ))
        position = scen.to_position_state(residual_edge=residual)
        action = eng.decide(
            position=position,
            exit_trigger=scen.exit_reason,
            current_edge=EdgeCandidate(
                date=scen.exit_date, ticker=scen.ticker,
                direction=0.05, conviction=0.7,
                raw_opportunity=scen.raw_opportunity,
                regime="neutral", regime_score=0.5,
                net_edge_5d=residual,
            ),
            replacement_candidates=[],
            signal_age_hours=14.0,  # KR↔US gap, fresh under V3.3
        )
        assert action.action_type == "KEEP"
        assert "thesis_alive" in action.reason


# ──────────────────────────────────────────────────────────────
# AMZN scenario — same fix
# ──────────────────────────────────────────────────────────────
class TestAMZNWinnerProtected:
    def test_v33_kept(self):
        scen = SCENARIOS["amzn_winner_cut"]
        residual = 0.003
        eng = ExitThesisEngine(policy=ExitPolicy(
            hold_min_residual_edge=0.0,
            reduce_zone_edge=0.001,
            max_signal_staleness_hours=16.0,
        ))
        position = scen.to_position_state(residual_edge=residual)
        action = eng.decide(
            position=position,
            exit_trigger=scen.exit_reason,
            current_edge=EdgeCandidate(
                date=scen.exit_date, ticker=scen.ticker,
                direction=0.04, conviction=0.6,
                raw_opportunity=scen.raw_opportunity,
                regime="neutral", regime_score=0.5,
                net_edge_5d=residual,
            ),
            replacement_candidates=[],
            signal_age_hours=14.0,
        )
        assert action.action_type == "KEEP"


# ──────────────────────────────────────────────────────────────
# FANG scenario — Pyramid winner add-on
# ──────────────────────────────────────────────────────────────
class TestFANGPyramidEligible:
    def test_pyramid_adds_to_undersized_winner(self):
        scen = SCENARIOS["fang_undersized_winner"]
        # Day 3 of holding, +2.6% pnl, residual still strong
        position = scen.to_position_state(residual_edge=0.005, current_tier="A")
        candidate = EdgeCandidate(
            date=scen.exit_date, ticker=scen.ticker,
            direction=0.06, conviction=0.75,
            raw_opportunity=0.005,
            regime="neutral", regime_score=0.5,
            net_edge_5d=0.005,
            edge_tier="A",
            win_probability_5d=0.65,
            expected_mae_5d=-0.020,
        )
        eng = PyramidPolicyEngine(policy=PyramidPolicy(
            min_unrealized_pnl_for_add=0.015,    # 1.5%
            min_residual_edge_for_add=0.004,
            min_current_tier_for_add="A",
            max_single_weight_after_add=0.40,
            add_size_fraction_of_initial=0.50,
        ))
        action = eng.evaluate(
            position=position,
            candidate=candidate,
            adds_so_far=0,
            initial_weight=scen.weight_at_entry,
        )
        assert action is not None
        assert action.action_type == "ADD_TO_WINNER"
        # Initial 0.18 + 0.18*0.5 = 0.27, capped at 0.40
        assert action.target_weight == pytest.approx(0.27)


# ──────────────────────────────────────────────────────────────
# Loss position — Pyramid invariant
# ──────────────────────────────────────────────────────────────
class TestLossPositionPyramidBlocked:
    def test_loss_position_never_pyramided(self):
        scen = SCENARIOS["loss_position_blocked_add"]
        position = scen.to_position_state(residual_edge=0.010, current_tier="S")
        # Even with very strong S-tier signal, loss position blocks pyramid
        candidate = EdgeCandidate(
            date=scen.exit_date, ticker=scen.ticker,
            direction=0.08, conviction=0.9,
            raw_opportunity=0.010,
            regime="strong_bull", regime_score=0.85,
            net_edge_5d=0.020,
            edge_tier="S",
            win_probability_5d=0.75,
            expected_mae_5d=-0.015,
        )
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(
            position=position,
            candidate=candidate,
            adds_so_far=0,
            initial_weight=scen.weight_at_entry,
        )
        # Forbid averaging-down invariant
        assert action is None


# ──────────────────────────────────────────────────────────────
# Scenario suite invariants
# ──────────────────────────────────────────────────────────────
class TestScenarioSuite:
    def test_documented_count(self):
        # 4 documented; full 14-trade extraction is a future PR
        assert len(SCENARIOS) == 4

    def test_all_have_unique_names(self):
        names = [s.name for s in SCENARIOS.values()]
        assert len(names) == len(set(names))

    def test_all_have_complete_data(self):
        for key, s in SCENARIOS.items():
            assert s.ticker
            assert s.entry_date < s.exit_date
            assert s.entry_price > 0
            assert s.exit_price > 0
            assert s.holding_days > 0
            assert s.weight_at_entry > 0
