"""Conditional entry filter — only trade when vol + direction align.

Medallion-inspired enhancements:
  1. Opportunity cost ranking (compare candidates by Sharpe-per-trade)
  2. Pre-entry liquidity check (volume, spread)
  3. Sector concentration limit (max 2 per GICS sector)
  4. Dynamic monthly trade limit (tighter in drawdown, looser when winning)
  5. Realistic expected profit with vol-adjusted slippage
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from v3.rules.direction import DirectionSignal


@dataclass
class EntryDecision:
    should_enter: bool
    ticker: str
    direction: str
    vol_score: float
    confidence: float
    direction_clarity: float
    opportunity_score: float   # Sharpe-per-trade ranking
    rejection_reason: str


class EntryFilter:
    """Decides whether to enter a position based on multiple conditions."""

    def __init__(
        self,
        min_direction_clarity: float = 0.3,
        max_trades_per_month: int = 5,
        max_positions: int = 3,
        min_vol_expansion: float = 0.1,
        min_confidence: float = 0.4,
        max_sector_concentration: int = 2,
        min_daily_volume_krw: float = 500_000_000,   # 5억 최소 거래대금
    ):
        self.min_clarity = min_direction_clarity
        self.base_max_monthly = max_trades_per_month
        self.max_positions = max_positions
        self.min_vol_expansion = min_vol_expansion
        self.min_confidence = min_confidence
        self.max_sector_conc = max_sector_concentration
        self.min_volume = min_daily_volume_krw

    def evaluate(
        self,
        ticker: str,
        vol_score: float,
        confidence: float,
        direction_signal: DirectionSignal,
        current_positions: int,
        monthly_trades: int,
        market: str = "KOSPI",
        cost_rate: float = 0.010,
        circuit_breaker_active: bool = False,
        recent_win_rate: float = 0.5,
        current_mdd: float = 0.0,
        ticker_volume: float = 0.0,
        sector: str = "",
        position_sectors: list[str] | None = None,
    ) -> EntryDecision:
        """Evaluate whether to enter a position with all Medallion-grade checks."""

        opp_score = self._opportunity_score(vol_score, confidence, direction_signal.clarity, cost_rate)

        # C1: Vol expansion predicted
        if vol_score < self.min_vol_expansion:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"vol_score {vol_score:.3f} < {self.min_vol_expansion}")

        # C2: Confidence sufficient
        if confidence < self.min_confidence:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"confidence {confidence:.3f} < {self.min_confidence}")

        # C3: Direction is clear enough
        if direction_signal.clarity < self.min_clarity:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"direction_clarity {direction_signal.clarity:.3f} < {self.min_clarity}")

        # C4: Direction is long (V3 long-only)
        if direction_signal.direction != "long":
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"direction={direction_signal.direction}")

        # C5: Dynamic monthly trade limit
        max_monthly = self._dynamic_monthly_limit(recent_win_rate, current_mdd)
        if monthly_trades >= max_monthly:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"monthly_trades {monthly_trades} >= dynamic_max {max_monthly}")

        # C6: Position capacity
        if current_positions >= self.max_positions:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"positions {current_positions} >= {self.max_positions}")

        # C7: Circuit breaker
        if circuit_breaker_active:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                "circuit_breaker_active")

        # C8: Expected profit > costs (vol-adjusted slippage)
        slippage_mult = 1.0 + vol_score * 0.2  # High vol → wider spreads
        expected_move = vol_score * min(confidence, 0.8) * direction_signal.clarity
        adjusted_cost = cost_rate * slippage_mult * 1.75   # Need 1.75x cost, not 1.5x
        if expected_move < adjusted_cost:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"expected {expected_move:.4f} < adjusted_cost {adjusted_cost:.4f}")

        # C9: Liquidity check
        if ticker_volume > 0 and ticker_volume < self.min_volume:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"volume {ticker_volume/1e6:.0f}M < min {self.min_volume/1e6:.0f}M")

        # C10: Sector concentration
        position_sectors = position_sectors or []
        if sector and position_sectors.count(sector) >= self.max_sector_conc:
            return self._reject(ticker, vol_score, confidence, direction_signal, opp_score,
                                f"sector {sector} already {self.max_sector_conc} positions")

        return EntryDecision(
            should_enter=True,
            ticker=ticker,
            direction="long",
            vol_score=vol_score,
            confidence=confidence,
            direction_clarity=direction_signal.clarity,
            opportunity_score=opp_score,
            rejection_reason="",
        )

    def rank_candidates(self, candidates: list[EntryDecision]) -> list[EntryDecision]:
        """Rank approved candidates by opportunity score (Sharpe-per-trade)."""
        return sorted(candidates, key=lambda c: c.opportunity_score, reverse=True)

    def _dynamic_monthly_limit(self, win_rate: float, mdd: float) -> int:
        """Adaptive trade limit: tighter in drawdown, looser when winning."""
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

    @staticmethod
    def _opportunity_score(vol_score: float, confidence: float, clarity: float, cost: float) -> float:
        """Expected risk-adjusted return per trade.

        Higher = better opportunity relative to risk.
        """
        expected_return = vol_score * confidence * clarity
        expected_risk = max(vol_score, 0.05)   # Vol itself is the risk
        net_return = expected_return - cost

        if expected_risk <= 0:
            return 0.0
        return net_return / expected_risk

    def _reject(
        self, ticker: str, vol_score: float, confidence: float,
        direction_signal: DirectionSignal, opp_score: float, reason: str,
    ) -> EntryDecision:
        return EntryDecision(
            should_enter=False,
            ticker=ticker,
            direction="skip",
            vol_score=vol_score,
            confidence=confidence,
            direction_clarity=direction_signal.clarity,
            opportunity_score=opp_score,
            rejection_reason=reason,
        )
