"""Signal generator — Phase 2 orchestrator (S4).

Pipeline:
  1. Compute directional alphas (trend, reversion, …)
  2. Compute conviction sources (vol, …)
  3. OpportunityScorer: direction × conviction → opportunity_score
  4. EntryFilter (operational constraints only)
  5. Sizer: allocate weights among approved candidates
  6. Emit TradeSignal

No mutations. No hidden state. No engine toggles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from v3.rules.entry import (
    EntryCandidate,
    EntryFilter,
    OperationalState,
)
from v3.strategy.alpha_sources import (
    compute_conviction,
    compute_directional,
    DEFAULT_CONVICTION,
    DEFAULT_DIRECTIONAL,
    AlphaSource,
    ConvictionSource,
)
from v3.strategy.opportunity import OpportunityReport, OpportunityScorer
from v3.strategy.regime_v2 import Regime
from v3.strategy.sizing import VolTargetSizer


@dataclass(frozen=True)
class TradeSignal:
    """Output of SignalGenerator.generate()."""

    action: str                            # "TRADE" or "CASH"
    positions: list[dict]                  # [{ticker, weight, direction, conviction, opportunity}]
    cash_weight: float
    regime_name: str
    regime_score: float
    position_scale: float
    n_candidates: int
    n_approved: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    opportunity_map: dict[str, float] = field(default_factory=dict)  # ticker → opportunity
    opportunity_gate: float = 0.0                                     # cost × gate_multiplier


class SignalGenerator:
    """Pure-function orchestrator for Phase 2 signal generation.

    Takes raw market data + a Regime snapshot, returns a TradeSignal.
    No configuration toggles — behavior flows from regime.alpha_weights.
    """

    def __init__(
        self,
        opportunity_scorer: OpportunityScorer,
        entry_filter: EntryFilter,
        sizer: VolTargetSizer,
        directional_sources: tuple[AlphaSource, ...] = DEFAULT_DIRECTIONAL,
        conviction_sources: tuple[ConvictionSource, ...] = DEFAULT_CONVICTION,
        max_positions: int = 3,
        max_single_weight: float = 0.40,
        min_position_weight: float = 0.02,
    ):
        self.scorer = opportunity_scorer
        self.entry = entry_filter
        self.sizer = sizer
        self.directional_sources = directional_sources
        self.conviction_sources = conviction_sources
        self.max_positions = max_positions
        self.max_weight = max_single_weight
        self.min_weight = min_position_weight

    def generate(
        self,
        ohlcv: pd.DataFrame,
        vol_scores: pd.DataFrame,
        regime: Regime,
        cost: float,
        state: OperationalState,
        ticker_volume_map: dict[str, float] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> TradeSignal:
        """Generate a TradeSignal.

        Args:
            ohlcv: Long-format OHLCV up to current date.
            vol_scores: DataFrame with [ticker, vol_score, confidence] (for conviction).
            regime: Immutable Regime snapshot (from RegimeDetectorV2.detect).
            cost: Round-trip transaction cost (e.g., 0.001 for NASDAQ).
            state: Operational state (positions, monthly count, drawdown, etc.).
            ticker_volume_map: optional {ticker: daily_volume_krw} for liquidity check.
            sector_map: optional {ticker: sector} for sector concentration.

        Returns:
            Immutable TradeSignal.
        """
        ticker_volume_map = ticker_volume_map or {}
        sector_map = sector_map or {}

        # Short circuit: bear regime always CASH
        if regime.name == "bear" or regime.position_scale <= 0:
            logger.info(f"Signal: CASH (regime={regime.name}, scale={regime.position_scale:.2f})")
            return TradeSignal(
                action="CASH", positions=[], cash_weight=1.0,
                regime_name=regime.name, regime_score=regime.score,
                position_scale=regime.position_scale,
                n_candidates=0, n_approved=0,
                rejection_reasons={"bear_regime": 0},
            )

        # 1+2. Compute alphas + conviction
        directional = compute_directional(ohlcv, sources=self.directional_sources)
        conviction = compute_conviction(
            ohlcv, vol_scores=vol_scores, sources=self.conviction_sources,
        )
        if directional.empty:
            logger.warning("Signal: no directional alphas — CASH")
            return TradeSignal(
                action="CASH", positions=[], cash_weight=1.0,
                regime_name=regime.name, regime_score=regime.score,
                position_scale=regime.position_scale,
                n_candidates=0, n_approved=0,
                rejection_reasons={"no_alphas": 0},
            )

        # 3. Opportunity scoring (single gate)
        report: OpportunityReport = self.scorer.score(
            directional_alphas=directional,
            conviction=conviction,
            regime=regime,
            cost=cost,
        )
        opportunity_approved = report.approved
        opp_map = {r.ticker: r.opportunity for r in report.rows}
        opp_gate = cost * self.scorer.gate_multiplier

        # 4. Operational filtering — pick best candidates that pass ops checks
        selected: list[EntryCandidate] = []
        rejections: dict[str, int] = {}

        for row in report.ranked():  # Already sorted by opportunity descending
            if not row.passes:
                # Below gate: stop (rest are lower opportunity)
                break
            if len(selected) >= self.max_positions:
                break

            candidate = EntryCandidate(
                ticker=row.ticker,
                opportunity=row.opportunity,
                direction=row.direction,
                conviction=row.conviction,
                ticker_volume_krw=float(ticker_volume_map.get(row.ticker, 0.0)),
                sector=sector_map.get(row.ticker, ""),
            )

            # Operational state reflects THIS candidate being a new position
            trial_state = OperationalState(
                current_positions=state.current_positions + len(selected),
                monthly_trades=state.monthly_trades + len(selected),
                recent_win_rate=state.recent_win_rate,
                current_mdd=state.current_mdd,
                circuit_breaker_active=state.circuit_breaker_active,
                existing_position_sectors=(
                    state.existing_position_sectors
                    + tuple(c.sector for c in selected if c.sector)
                ),
            )
            decision = self.entry.check(candidate, trial_state)
            if decision.should_enter:
                selected.append(candidate)
            else:
                key = decision.reason.split(" ")[0] if decision.reason else "unknown"
                rejections[key] = rejections.get(key, 0) + 1

        # 5. Sizing among selected
        if not selected:
            logger.info(
                f"Signal: CASH — opportunity={len(opportunity_approved)}, "
                f"approved=0, regime={regime.name}, rejections={rejections}"
            )
            return TradeSignal(
                action="CASH", positions=[], cash_weight=1.0,
                regime_name=regime.name, regime_score=regime.score,
                position_scale=regime.position_scale,
                n_candidates=len(report.rows),
                n_approved=len(opportunity_approved),
                rejection_reasons=rejections,
                opportunity_map=opp_map,
                opportunity_gate=opp_gate,
            )

        positions = self._size(selected, regime.position_scale, ohlcv, vol_scores)

        logger.info(
            f"Signal: TRADE — {len(positions)} positions, "
            f"regime={regime.name} (score={regime.score:.2f}, "
            f"scale={regime.position_scale:.2f}), "
            f"opportunity_approved={len(opportunity_approved)}, "
            f"rejections={rejections}"
        )

        total_weight = sum(p["weight"] for p in positions)
        return TradeSignal(
            action="TRADE",
            positions=positions,
            cash_weight=max(0.0, 1.0 - total_weight),
            regime_name=regime.name,
            regime_score=regime.score,
            position_scale=regime.position_scale,
            n_candidates=len(report.rows),
            n_approved=len(opportunity_approved),
            rejection_reasons=rejections,
            opportunity_map=opp_map,
            opportunity_gate=opp_gate,
        )

    # ── private ───────────────────────────────────────────
    def _size(
        self,
        selected: list[EntryCandidate],
        position_scale: float,
        ohlcv: pd.DataFrame,
        vol_scores: pd.DataFrame | None,
    ) -> list[dict]:
        """Apply VolTargetSizer + regime position_scale + min/max clamps."""
        # Build predicted vol & confidence maps for sizer
        if vol_scores is None or vol_scores.empty:
            # Default: equal vol
            predicted_vols = {c.ticker: 0.20 for c in selected}
            confidences = {c.ticker: c.conviction for c in selected}
        else:
            vs = vol_scores.set_index("ticker")
            predicted_vols = {
                c.ticker: max(float(vs.loc[c.ticker, "vol_score"])
                              if c.ticker in vs.index else 0.20, 0.05)
                for c in selected
            }
            confidences = {c.ticker: c.conviction for c in selected}

        raw_weights = self.sizer.size_portfolio(predicted_vols, confidences)

        positions: list[dict] = []
        for c in selected:
            w = raw_weights.get(c.ticker, 1.0 / max(len(selected), 1))
            w = round(w * position_scale, 4)
            w = min(w, self.max_weight)
            if w < self.min_weight:
                continue
            positions.append({
                "ticker": c.ticker,
                "weight": w,
                "direction": round(c.direction, 4),
                "conviction": round(c.conviction, 4),
                "opportunity": round(c.opportunity, 5),
            })

        # Normalize if overshooting 95%
        total = sum(p["weight"] for p in positions)
        if total > 0.95:
            scale = 0.95 / total
            for p in positions:
                p["weight"] = round(p["weight"] * scale, 4)

        return positions
