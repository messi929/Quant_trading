"""Operational entry filter (Phase 2 S4).

This file replaces the Phase 1 multi-condition EntryFilter. The alpha/vol/
direction gates are now handled by OpportunityScorer (v3/strategy/opportunity.py).
This module only enforces OPERATIONAL constraints that are orthogonal to alpha:

  - Position capacity (max concurrent positions)
  - Monthly trade limit (dynamic, based on drawdown/win rate)
  - Circuit breaker halt
  - Liquidity minimum (daily dollar volume)
  - Sector concentration cap

Design:
  - Pure checks (no mutation of candidate data)
  - Explicit decision object with rejection reason
  - Single method `check(candidate, state)` — no hidden state

The Phase 1 EntryFilter class is preserved (same name) but its behavior is
reduced to these operational checks. Downstream callers (signal.py) first run
OpportunityScorer, then pass approved candidates through EntryFilter for
operational gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class EntryCandidate:
    """Immutable input to operational filter (produced by OpportunityScorer)."""

    ticker: str
    opportunity: float
    direction: float
    conviction: float
    ticker_volume_krw: float = 0.0   # daily dollar volume (for liquidity check)
    sector: str = ""                  # GICS/sector string (optional)


@dataclass(frozen=True)
class OperationalState:
    """State snapshot passed to `EntryFilter.check`."""

    current_positions: int
    monthly_trades: int
    recent_win_rate: float = 0.5
    current_mdd: float = 0.0
    circuit_breaker_active: bool = False
    existing_position_sectors: tuple[str, ...] = ()


class EntryDecision(NamedTuple):
    """Immutable decision with rejection reason."""

    should_enter: bool
    ticker: str
    reason: str


class EntryFilter:
    """Operational constraint gate (no alpha logic).

    Alpha / direction / vol checks are handled upstream by OpportunityScorer.
    This filter only protects against operational violations (exposure,
    liquidity, etc.).
    """

    def __init__(
        self,
        max_positions: int = 3,
        max_trades_per_month: int = 5,
        min_daily_volume_krw: float = 500_000_000,
        max_sector_concentration: int = 2,
    ):
        if max_positions < 1:
            raise ValueError(f"max_positions must be >= 1: {max_positions}")
        if max_trades_per_month < 1:
            raise ValueError(f"max_trades_per_month must be >= 1: {max_trades_per_month}")
        self.max_positions = max_positions
        self.base_max_monthly = max_trades_per_month
        self.min_volume = min_daily_volume_krw
        self.max_sector_conc = max_sector_concentration

    def check(
        self,
        candidate: EntryCandidate,
        state: OperationalState,
    ) -> EntryDecision:
        """Validate operational constraints. Pure function."""

        # C1: Circuit breaker halt
        if state.circuit_breaker_active:
            return EntryDecision(False, candidate.ticker, "circuit_breaker_active")

        # C2: Position capacity
        if state.current_positions >= self.max_positions:
            return EntryDecision(
                False, candidate.ticker,
                f"positions {state.current_positions} >= {self.max_positions}",
            )

        # C3: Monthly trade limit (dynamic)
        max_monthly = self._dynamic_monthly_limit(
            win_rate=state.recent_win_rate, mdd=state.current_mdd
        )
        if state.monthly_trades >= max_monthly:
            return EntryDecision(
                False, candidate.ticker,
                f"monthly_trades {state.monthly_trades} >= dynamic_max {max_monthly}",
            )

        # C4: Liquidity (skip if not provided)
        if candidate.ticker_volume_krw > 0 and candidate.ticker_volume_krw < self.min_volume:
            return EntryDecision(
                False, candidate.ticker,
                f"volume {candidate.ticker_volume_krw/1e6:.0f}M < "
                f"min {self.min_volume/1e6:.0f}M",
            )

        # C5: Sector concentration
        if candidate.sector and (
            state.existing_position_sectors.count(candidate.sector)
            >= self.max_sector_conc
        ):
            return EntryDecision(
                False, candidate.ticker,
                f"sector {candidate.sector} already at "
                f"{self.max_sector_conc} positions",
            )

        return EntryDecision(True, candidate.ticker, "")

    def _dynamic_monthly_limit(self, win_rate: float, mdd: float) -> int:
        """Adaptive monthly trade cap: tighter in drawdown, looser when winning."""
        limit = self.base_max_monthly
        if mdd > 0.10:
            limit = max(2, limit - 2)
        elif mdd > 0.05:
            limit = max(3, limit - 1)
        if win_rate < 0.40:
            limit = max(2, limit - 1)
        elif win_rate > 0.65:
            limit = min(8, limit + 2)
        return limit
