"""Combined signal: vol prediction + direction rules → trade signals.

Integrates VolTargetSizer for covariance-aware Kelly sizing,
passes regime to direction engine, and applies regime scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from v3.rules.direction import DirectionEngine, DirectionSignal
from v3.rules.entry import EntryFilter, EntryDecision
from v3.strategy.sizing import VolTargetSizer
from v3.strategy.regime import RegimeState


@dataclass
class TradeSignal:
    action: str               # "TRADE" or "CASH"
    positions: list[dict]     # [{"ticker", "weight", "vol_score", "confidence", "direction"}]
    cash_weight: float
    regime: str
    n_candidates: int
    n_rejected: int
    rejection_reasons: dict


class SignalGenerator:
    """Generates V3 trade signals with full pipeline integration."""

    def __init__(
        self,
        direction_engine: DirectionEngine,
        entry_filter: EntryFilter,
        sizer: VolTargetSizer,
        max_positions: int = 3,
        max_single_weight: float = 0.40,
    ):
        self.direction = direction_engine
        self.entry = entry_filter
        self.sizer = sizer
        self.max_positions = max_positions
        self.max_weight = max_single_weight

    def generate(
        self,
        vol_scores: pd.DataFrame,
        full_data: pd.DataFrame,
        regime_state: RegimeState | None = None,
        event_scores: dict[str, float] | None = None,
        current_positions: int = 0,
        monthly_trades: int = 0,
        market_cost: float = 0.010,
        circuit_breaker: bool = False,
        recent_win_rate: float = 0.5,
        current_mdd: float = 0.0,
    ) -> TradeSignal:
        """Generate trade signal with regime-aware direction and proper sizing."""
        regime = regime_state.regime if regime_state else "neutral"
        regime_scale = regime_state.scale_factor if regime_state else 1.0

        # Bear regime → CASH immediately
        if regime == "bear" and regime_scale == 0.0:
            return TradeSignal(
                action="CASH", positions=[], cash_weight=1.0,
                regime=regime, n_candidates=len(vol_scores),
                n_rejected=len(vol_scores), rejection_reasons={"bear_regime": len(vol_scores)},
            )

        # Step 1: Direction evaluation with regime
        direction_signals = self.direction.evaluate_batch(full_data, event_scores, regime=regime)

        # Step 2: Entry filter with all checks
        approved = []
        rejections = {}
        vol_sorted = vol_scores.sort_values("vol_score", ascending=False)

        for _, row in vol_sorted.iterrows():
            ticker = row["ticker"]
            vol_score = row["vol_score"]
            confidence = row["confidence"]

            dir_signal = direction_signals.get(ticker)
            if dir_signal is None:
                continue

            market = self._detect_market(ticker)
            cost = market_cost if market in ("KOSPI", "KOSDAQ") else 0.001

            # Estimate daily trading value for liquidity check
            ticker_data = full_data[full_data["ticker"] == ticker]
            ticker_volume = 0.0
            if not ticker_data.empty:
                recent = ticker_data.tail(5)
                avg_vol = recent["volume"].mean()
                avg_price = recent["close"].mean()
                ticker_volume = avg_vol * avg_price

            decision = self.entry.evaluate(
                ticker=ticker,
                vol_score=vol_score,
                confidence=confidence,
                direction_signal=dir_signal,
                current_positions=current_positions + len(approved),
                monthly_trades=monthly_trades + len(approved),
                market=market,
                cost_rate=cost,
                circuit_breaker_active=circuit_breaker,
                recent_win_rate=recent_win_rate,
                current_mdd=current_mdd,
                ticker_volume=ticker_volume,
            )

            if decision.should_enter:
                approved.append({
                    "ticker": ticker,
                    "vol_score": vol_score,
                    "confidence": confidence,
                    "direction": decision.direction,
                    "clarity": decision.direction_clarity,
                    "opportunity_score": decision.opportunity_score,
                    "market": market,
                })
            else:
                # Keep full rejection reason for debugging
                reason_key = decision.rejection_reason.split(" ")[0] if decision.rejection_reason else "unknown"
                rejections[reason_key] = rejections.get(reason_key, 0) + 1

            if len(approved) >= self.max_positions:
                break

        # Rank by opportunity score (highest first)
        ranked = sorted(approved, key=lambda a: a.get("opportunity_score", 0), reverse=True)

        if not ranked:
            return TradeSignal(
                action="CASH", positions=[], cash_weight=1.0,
                regime=regime, n_candidates=len(vol_scores),
                n_rejected=len(vol_scores), rejection_reasons=rejections,
            )

        # Step 3: Size with VolTargetSizer (covariance + Kelly)
        positions = self._size_with_vol_target(ranked, regime_scale)

        logger.info(f"Signal: TRADE — {len(positions)} positions, "
                     f"regime={regime}(×{regime_scale:.1f}), "
                     f"rejected={sum(rejections.values())}")

        return TradeSignal(
            action="TRADE",
            positions=positions,
            cash_weight=1.0 - sum(p["weight"] for p in positions),
            regime=regime,
            n_candidates=len(vol_scores),
            n_rejected=len(vol_scores) - len(ranked),
            rejection_reasons=rejections,
        )

    def _size_with_vol_target(self, candidates: list[dict], regime_scale: float) -> list[dict]:
        """Size using VolTargetSizer with regime scaling."""
        predicted_vols = {c["ticker"]: max(c["vol_score"], 0.05) for c in candidates}
        confidences = {c["ticker"]: c["confidence"] for c in candidates}

        # Portfolio sizing with correlation
        weights = self.sizer.size_portfolio(predicted_vols, confidences)

        # Apply regime scale
        positions = []
        for c in candidates:
            weight = weights.get(c["ticker"], 0.05)
            weight = round(weight * regime_scale, 4)
            weight = min(weight, self.max_weight)

            if weight < 0.02:  # Skip tiny positions
                continue

            positions.append({
                "ticker": c["ticker"],
                "weight": weight,
                "vol_score": round(c["vol_score"], 4),
                "confidence": round(c["confidence"], 4),
                "direction": c["direction"],
            })

        # Normalize if total > 0.95
        total = sum(p["weight"] for p in positions)
        if total > 0.95:
            for p in positions:
                p["weight"] = round(p["weight"] / total * 0.95, 4)

        return positions

    @staticmethod
    def _detect_market(ticker: str) -> str:
        if ticker.endswith(".KS"):
            return "KOSPI"
        elif ticker.endswith(".KQ"):
            return "KOSDAQ"
        elif ticker.split(".")[0].isdigit():
            return "KOSPI"
        return "NASDAQ"
