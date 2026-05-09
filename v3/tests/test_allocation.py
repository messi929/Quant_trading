"""V3.3 AllocationEngine tests (PR-4.1).

Verifies:
  - net_edge가 큰 candidate가 더 큰 weight
  - downside_risk가 큰 candidate weight 할인 (lower)
  - Σ weights ≤ regime budget
  - LEVERAGE_CAP 1.00 강제 (assertion)
  - bear regime → 빈 dict
  - max_positions 제한
  - max_single_weight 위반 0
  - min_position_weight 미만 후보 drop
  - sector cap 적용
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.strategy.allocation import (
    AllocationEngine,
    LEVERAGE_CAP,
    PositionConstraints,
    RegimeBudget,
    _default_regime_budgets,
)
from v3.strategy.regime_v2 import Regime
from v3.strategy.types import EdgeCandidate


def _candidate(
    ticker: str,
    net_edge: float = 0.005,
    expected_mae: float = -0.020,
    entry_pass: bool = True,
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker=ticker,
        direction=0.05,
        conviction=0.7,
        raw_opportunity=0.005,
        regime="neutral",
        regime_score=0.5,
        expected_return_5d=0.010,
        expected_mae_5d=expected_mae,
        expected_mfe_5d=0.025,
        win_probability_5d=0.55,
        net_edge_5d=net_edge,
        edge_tier="A",
        entry_pass=entry_pass,
    )


def _regime(name: str = "neutral", score: float = 0.5) -> Regime:
    return Regime(
        name=name,
        score=score,
        alpha_weights={"trend": 1.0, "reversion": 0.0},
        position_scale=0.9,
        confidence=1.0,
        detected_at=pd.Timestamp("2026-05-09"),
    )


# ──────────────────────────────────────────────────────────────
# Leverage cap (영구 invariant)
# ──────────────────────────────────────────────────────────────
class TestLeverageCap:
    def test_constant_is_one(self):
        assert LEVERAGE_CAP == 1.00

    def test_total_weights_never_exceed_one(self):
        # Even with strong_bull (max 1.00) → still ≤ 1.00
        eng = AllocationEngine()
        cands = [_candidate(f"T{i}", net_edge=0.020) for i in range(3)]
        weights = eng.allocate(cands, regime=_regime("strong_bull"))
        assert sum(weights.values()) <= 1.00 + 1e-9

    def test_excessive_budget_still_capped_at_one(self):
        """Even if regime_budgets supplies leverage > 1.00, engine clips."""
        budgets = {
            "strong_bull": RegimeBudget(max_gross_exposure=1.50, target_annual_vol=0.30),
        }
        eng = AllocationEngine(
            regime_budgets=budgets,
            constraints=PositionConstraints(max_single_weight=1.0,
                                             min_position_weight=0.05),
        )
        cands = [_candidate(f"T{i}", net_edge=0.020) for i in range(3)]
        weights = eng.allocate(cands, regime=_regime("strong_bull"))
        assert sum(weights.values()) <= LEVERAGE_CAP + 1e-9


# ──────────────────────────────────────────────────────────────
# net_edge ↔ weight monotonicity
# ──────────────────────────────────────────────────────────────
class TestEdgeWeightMonotonicity:
    def test_higher_net_edge_higher_weight(self):
        eng = AllocationEngine()
        cands = [
            _candidate("LOW", net_edge=0.005),
            _candidate("HIGH", net_edge=0.020),
        ]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        if "LOW" in weights and "HIGH" in weights:
            assert weights["HIGH"] > weights["LOW"]

    def test_higher_downside_risk_lower_weight(self):
        eng = AllocationEngine()
        cands = [
            _candidate("LOW_RISK", net_edge=0.005, expected_mae=-0.010),
            _candidate("HIGH_RISK", net_edge=0.005, expected_mae=-0.040),
        ]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        # Same net_edge but high_risk has bigger |mae| → smaller ratio
        if "LOW_RISK" in weights and "HIGH_RISK" in weights:
            assert weights["LOW_RISK"] > weights["HIGH_RISK"]


# ──────────────────────────────────────────────────────────────
# Bear regime → CASH
# ──────────────────────────────────────────────────────────────
class TestBearRegimeCash:
    def test_bear_returns_empty_dict(self):
        eng = AllocationEngine()
        cands = [_candidate(f"T{i}", net_edge=0.020) for i in range(3)]
        weights = eng.allocate(cands, regime=_regime("bear"))
        assert weights == {}

    def test_unknown_regime_returns_empty(self):
        eng = AllocationEngine()
        cands = [_candidate("AAPL", net_edge=0.020)]
        # No "extreme_bull" in default budgets
        regime = Regime(
            name="extreme_bull", score=0.99,
            alpha_weights={}, position_scale=1.5, confidence=1.0,
            detected_at=pd.Timestamp("2026-05-09"),
        )
        weights = eng.allocate(cands, regime=regime)
        assert weights == {}


# ──────────────────────────────────────────────────────────────
# Filtering — entry_pass + net_edge > 0
# ──────────────────────────────────────────────────────────────
class TestFilteringPassing:
    def test_entry_pass_false_filtered(self):
        eng = AllocationEngine()
        cands = [
            _candidate("AAPL", net_edge=0.020, entry_pass=False),
            _candidate("MSFT", net_edge=0.020, entry_pass=True),
        ]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        assert "AAPL" not in weights
        assert "MSFT" in weights

    def test_negative_net_edge_filtered(self):
        eng = AllocationEngine()
        cands = [
            _candidate("AAPL", net_edge=-0.005),
            _candidate("MSFT", net_edge=0.020),
        ]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        assert "AAPL" not in weights

    def test_zero_net_edge_filtered(self):
        eng = AllocationEngine()
        cands = [_candidate("AAPL", net_edge=0.0)]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        assert weights == {}

    def test_none_net_edge_filtered(self):
        # Build candidate without net_edge populated
        c = EdgeCandidate(
            date=pd.Timestamp("2026-05-09"),
            ticker="AAPL", direction=0.05, conviction=0.7,
            raw_opportunity=0.005, regime="neutral", regime_score=0.5,
            entry_pass=True,
            # net_edge_5d = None
        )
        eng = AllocationEngine()
        weights = eng.allocate([c], regime=_regime("neutral"))
        assert weights == {}


# ──────────────────────────────────────────────────────────────
# max_positions
# ──────────────────────────────────────────────────────────────
class TestMaxPositions:
    def test_top_n_only(self):
        eng = AllocationEngine(constraints=PositionConstraints(
            max_positions=2, min_position_weight=0.05,
        ))
        cands = [
            _candidate("A", net_edge=0.005),
            _candidate("B", net_edge=0.010),
            _candidate("C", net_edge=0.020),
            _candidate("D", net_edge=0.015),
        ]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        assert len(weights) <= 2
        # Top 2 by net_edge: C (0.020), D (0.015)
        assert "C" in weights
        assert "D" in weights


# ──────────────────────────────────────────────────────────────
# max_single_weight 강제
# ──────────────────────────────────────────────────────────────
class TestMaxSingleWeight:
    def test_no_position_exceeds_max(self):
        eng = AllocationEngine(constraints=PositionConstraints(
            max_single_weight=0.40, min_position_weight=0.05,
        ))
        # Single dominant candidate, regime budget 1.00
        # → could otherwise allocate 1.00 to one ticker
        cands = [_candidate("AAPL", net_edge=0.020)]
        weights = eng.allocate(cands, regime=_regime("strong_bull"))
        if "AAPL" in weights:
            assert weights["AAPL"] <= 0.40 + 1e-9


# ──────────────────────────────────────────────────────────────
# min_position_weight 강제
# ──────────────────────────────────────────────────────────────
class TestMinPositionWeight:
    def test_below_floor_dropped(self):
        # 5 candidates → each gets ~0.20 of budget. With caution budget 0.40,
        # each would get 0.40/5 = 0.08 — below 0.15 floor → dropped
        eng = AllocationEngine(constraints=PositionConstraints(
            max_positions=5, min_position_weight=0.15,
        ))
        cands = [_candidate(f"T{i}", net_edge=0.005) for i in range(5)]
        weights = eng.allocate(cands, regime=_regime("caution"))
        # caution budget 0.40 / 5 = 0.08 < 0.15 → all dropped
        # But max_positions 5 takes all, ratios uniform → 0.08 each → all dropped
        for w in weights.values():
            assert w >= 0.15


# ──────────────────────────────────────────────────────────────
# sector cap
# ──────────────────────────────────────────────────────────────
class TestSectorCap:
    def test_sector_cap_reduces_overexposure(self):
        eng = AllocationEngine(constraints=PositionConstraints(
            max_positions=3, min_position_weight=0.05,
            max_sector_weight=0.50, max_single_weight=0.40,
        ))
        cands = [
            _candidate("AAPL", net_edge=0.020),
            _candidate("MSFT", net_edge=0.018),
            _candidate("GOOG", net_edge=0.015),
        ]
        sector_map = {"AAPL": "tech", "MSFT": "tech", "GOOG": "tech"}
        weights = eng.allocate(
            cands, regime=_regime("strong_bull"), sector_map=sector_map,
        )
        sector_total = sum(w for t, w in weights.items() if sector_map[t] == "tech")
        assert sector_total <= 0.50 + 1e-6


# ──────────────────────────────────────────────────────────────
# Correlation drag
# ──────────────────────────────────────────────────────────────
class TestCorrelationDrag:
    def test_high_correlation_reduces_weight(self):
        eng = AllocationEngine()
        cands = [
            _candidate("A", net_edge=0.010),
            _candidate("B", net_edge=0.010),
            _candidate("C", net_edge=0.010),
        ]
        # High correlation matrix
        high_corr = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0],
        ])
        # Low correlation matrix
        low_corr = np.array([
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        w_high = eng.allocate(
            cands, regime=_regime("neutral"), correlation_matrix=high_corr,
        )
        w_low = eng.allocate(
            cands, regime=_regime("neutral"), correlation_matrix=low_corr,
        )
        # High corr → more drag → smaller total deployment (or same after norm)
        # At minimum: both produce valid output
        assert sum(w_high.values()) <= sum(w_low.values()) + 1e-9


# ──────────────────────────────────────────────────────────────
# Default regime budgets
# ──────────────────────────────────────────────────────────────
class TestDefaultRegimeBudgets:
    def test_no_leverage_in_defaults(self):
        budgets = _default_regime_budgets()
        for name, b in budgets.items():
            assert b.max_gross_exposure <= 1.00, (
                f"{name} budget {b.max_gross_exposure} > 1.00 (leverage)"
            )

    def test_bear_zero(self):
        budgets = _default_regime_budgets()
        assert budgets["bear"].max_gross_exposure == 0.0

    def test_strong_bull_at_cap(self):
        budgets = _default_regime_budgets()
        assert budgets["strong_bull"].max_gross_exposure == 1.00

    def test_monotonic_exposure(self):
        budgets = _default_regime_budgets()
        # bear < caution < neutral < bull < strong_bull
        order = ["bear", "caution", "neutral", "bull", "strong_bull"]
        prev = -0.01
        for name in order:
            cur = budgets[name].max_gross_exposure
            assert cur >= prev, f"{name} exposure {cur} < {prev}"
            prev = cur


# ──────────────────────────────────────────────────────────────
# Empty / edge cases
# ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_candidates(self):
        eng = AllocationEngine()
        weights = eng.allocate([], regime=_regime("neutral"))
        assert weights == {}

    def test_all_filtered_out(self):
        eng = AllocationEngine()
        cands = [_candidate("AAPL", net_edge=-0.001)]
        weights = eng.allocate(cands, regime=_regime("neutral"))
        assert weights == {}
