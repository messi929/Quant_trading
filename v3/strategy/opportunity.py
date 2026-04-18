"""Opportunity scorer — single source of truth for entry decisions (Phase 2 S4).

Formula (from Phase 2 principle #1 "Single source of truth"):

  direction(ticker)   = Σ_a  w_a(regime) · α_a(ticker)       ∈ [-0.1, 0.1]
  conviction(ticker)  = Π_c  c_s(ticker)                      ∈ [0, 1]
  opportunity(ticker) = direction · conviction                ∈ [-0.1, 0.1]

  enter_if  opportunity > cost × k    (k ≈ 1.75)

Design:
  - Pure function (no state, no mutation)
  - No dependency on EntryFilter internals — opportunity is the single gate
  - Transparent: per-ticker breakdown for logging/audit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from loguru import logger

from v3.strategy.regime_v2 import Regime


class OpportunityRow(NamedTuple):
    """Per-ticker opportunity breakdown for observability."""

    ticker: str
    direction: float       # Σ w·α
    conviction: float      # Π c
    opportunity: float     # direction × conviction
    gate_value: float      # cost × k
    passes: bool           # opportunity > gate_value


@dataclass(frozen=True)
class OpportunityReport:
    """Immutable scorecard produced by OpportunityScorer.score()."""

    rows: list[OpportunityRow]
    regime: Regime
    cost: float
    gate_multiplier: float

    @property
    def approved(self) -> list[OpportunityRow]:
        """Rows that pass the opportunity gate (opportunity > cost × k)."""
        return [r for r in self.rows if r.passes]

    def top(self, n: int = 10) -> list[OpportunityRow]:
        """Top N rows by opportunity (descending)."""
        return sorted(self.rows, key=lambda r: r.opportunity, reverse=True)[:n]

    def ranked(self) -> list[OpportunityRow]:
        """All rows sorted by opportunity descending (for selection)."""
        return sorted(self.rows, key=lambda r: r.opportunity, reverse=True)


class OpportunityScorer:
    """Combines directional alphas + conviction sources into an entry score.

    Stateless. `score()` is a pure function of its inputs.
    """

    def __init__(self, gate_multiplier: float = 1.75):
        if gate_multiplier <= 0:
            raise ValueError(f"gate_multiplier must be > 0: {gate_multiplier}")
        self.gate_multiplier = gate_multiplier

    def score(
        self,
        directional_alphas: pd.DataFrame,
        conviction: pd.DataFrame,
        regime: Regime,
        cost: float,
    ) -> OpportunityReport:
        """Compute per-ticker opportunity.

        Args:
            directional_alphas: ticker-indexed DF, columns = alpha names, values in [-0.1, 0.1].
            conviction: ticker-indexed DF, columns = conviction source names, values in [0, 1].
            regime: Regime snapshot (provides alpha_weights).
            cost: Round-trip transaction cost (e.g., 0.001 for NASDAQ).

        Returns:
            OpportunityReport containing per-ticker rows + metadata.
        """
        if directional_alphas is None or directional_alphas.empty:
            return OpportunityReport(
                rows=[], regime=regime, cost=cost,
                gate_multiplier=self.gate_multiplier,
            )

        weights = regime.alpha_weights or {}

        # Align alpha columns with regime weights (intersection)
        usable_alphas = [a for a in weights if a in directional_alphas.columns]
        if not usable_alphas:
            logger.warning(
                "OpportunityScorer: no overlap between regime.alpha_weights "
                f"{list(weights.keys())} and alpha columns "
                f"{list(directional_alphas.columns)}"
            )
            return OpportunityReport(
                rows=[], regime=regime, cost=cost,
                gate_multiplier=self.gate_multiplier,
            )

        # Direction: weighted sum of alphas
        # directional_alphas has rows = tickers, cols = alphas
        weight_vec = pd.Series({a: weights[a] for a in usable_alphas})
        direction_series = directional_alphas[usable_alphas].mul(weight_vec, axis=1).sum(axis=1)

        # Conviction: product across all conviction sources (empty → 1.0 neutral)
        if conviction is None or conviction.empty:
            conviction_series = pd.Series(1.0, index=direction_series.index)
        else:
            aligned = conviction.reindex(direction_series.index).fillna(1.0)
            conviction_series = aligned.prod(axis=1)

        # Gate
        gate_value = cost * self.gate_multiplier

        # Build rows
        rows: list[OpportunityRow] = []
        for ticker in direction_series.index:
            d = float(direction_series.loc[ticker])
            c = float(conviction_series.loc[ticker]) if ticker in conviction_series.index else 1.0
            opp = d * c
            rows.append(OpportunityRow(
                ticker=str(ticker),
                direction=d,
                conviction=c,
                opportunity=opp,
                gate_value=gate_value,
                passes=opp > gate_value,
            ))

        n_pass = sum(1 for r in rows if r.passes)
        logger.info(
            f"Opportunity: {len(rows)} tickers, {n_pass} pass gate "
            f"(gate={gate_value:.5f}, regime={regime.name}, "
            f"weights={weights})"
        )

        return OpportunityReport(
            rows=rows, regime=regime, cost=cost,
            gate_multiplier=self.gate_multiplier,
        )


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime

    # Synthetic: 10 tickers
    tickers = [f"T{i:02d}" for i in range(10)]
    directional = pd.DataFrame({
        "trend":     [0.08, 0.05, 0.02, -0.01, -0.03, 0.06, 0.04, -0.02, 0.07, 0.00],
        "reversion": [-0.03, 0.02, 0.05, 0.04, -0.01, -0.02, -0.03, 0.06, -0.04, 0.03],
    }, index=tickers)
    conviction = pd.DataFrame({
        "vol": [0.9, 0.7, 0.3, 0.6, 0.5, 0.8, 0.4, 0.7, 0.85, 0.5],
    }, index=tickers)

    regime = Regime(
        name="neutral", score=0.5,
        alpha_weights={"trend": 1.0, "reversion": 0.0},
        position_scale=0.8, confidence=1.0,
        detected_at=pd.Timestamp(datetime.now().date()),
    )

    scorer = OpportunityScorer(gate_multiplier=1.75)
    report = scorer.score(
        directional_alphas=directional,
        conviction=conviction,
        regime=regime,
        cost=0.001,  # NASDAQ
    )

    print("=== Opportunity Report ===")
    print(f"Regime: {regime.describe()}")
    print(f"Gate: {report.rows[0].gate_value:.5f}")
    print()
    print(f"{'Ticker':<8} {'Direction':>10} {'Conviction':>10} {'Opportunity':>12} {'Passes':>7}")
    for r in report.ranked():
        print(f"{r.ticker:<8} {r.direction:>10.4f} {r.conviction:>10.2f} "
              f"{r.opportunity:>12.5f} {str(r.passes):>7}")

    print(f"\nApproved: {len(report.approved)}")
    assert len(report.rows) == 10
    # Top by opportunity should pass: T00 has direction 0.08 × conviction 0.9 = 0.072 > 0.00175
    assert report.top(1)[0].passes, "Top ticker should pass the gate"
    print("Smoke test passed.")
