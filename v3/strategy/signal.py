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

        # Sizer가 모든 candidate를 drop한 경우 (전체 floor 미달 또는
        # position_scale < sizer.min_weight). 현재 config에서는 도달 불가
        # (POSITION_SCALE_CURVE 최저 0.30 > floor 0.15)이나 구조적 가드.
        if not positions:
            rejections["sizer_dropped_all"] = rejections.get("sizer_dropped_all", 0) + len(selected)
            logger.info(
                f"Signal: CASH — sizer dropped all {len(selected)} selected "
                f"(position_scale={regime.position_scale:.2f} vs "
                f"sizer.min_weight={self.sizer.min_weight}), regime={regime.name}"
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
        vol_scores: pd.DataFrame | None,  # noqa: ARG002 — kept for signature stability
    ) -> list[dict]:
        """Size approved candidates with portfolio-level exposure cap.

        position_scale은 종목당 곱셈이 아니라 portfolio max_gross_exposure 한도
        (V3.3 RegimeBudget과 정합). 종목당 floor(sizer.min_weight)는 절대값으로
        유지. 총합이 한도 초과 시 opportunity 약한 종목부터 drop. 페르소나
        ②"1~3종목 집중, 들어가면 의미있게" + ①"확신 없으면 현금" 보장.

        predicted_vol은 ohlcv['vol_cc_20d'] (annualized, feature_engineer.py:99)
        에서 가져옴. vol_scores['vol_score']는 VolTransformer prediction signal
        이며 annualized volatility가 아니므로 sizing에 직접 쓰면 안 됨.
        """
        predicted_vols = self._extract_vols(selected, ohlcv)
        confidences = {c.ticker: c.conviction for c in selected}

        raw_weights = self.sizer.size_portfolio(predicted_vols, confidences)

        # Stage 1: per-position cap + floor filter
        candidate_map = {c.ticker: c for c in selected}
        weights: dict[str, float] = {}
        for c in selected:
            w = raw_weights.get(c.ticker, 1.0 / max(len(selected), 1))
            w = min(w, self.max_weight)
            if w < self.sizer.min_weight:
                # 의미 있는 사이즈 안 나오면 진입 안 함
                continue
            weights[c.ticker] = round(w, 4)

        # Stage 2: portfolio exposure cap — drop weakest until total ≤ scale
        dropped_for_exposure: list[tuple[str, float]] = []
        while weights and sum(weights.values()) > position_scale + 1e-6:
            weakest = min(
                weights.keys(),
                key=lambda t: candidate_map[t].opportunity,
            )
            dropped_for_exposure.append((weakest, weights[weakest]))
            del weights[weakest]
        if dropped_for_exposure:
            logger.info(
                f"Sizer exposure cap: dropped {len(dropped_for_exposure)} weakest "
                f"to fit position_scale={position_scale:.3f} — "
                f"{[(t, round(w, 4)) for t, w in dropped_for_exposure]}"
            )

        return [
            {
                "ticker": t,
                "weight": weights[t],
                "direction": round(candidate_map[t].direction, 4),
                "conviction": round(candidate_map[t].conviction, 4),
                "opportunity": round(candidate_map[t].opportunity, 5),
            }
            for t in weights
        ]

    def _extract_vols(
        self,
        selected: list[EntryCandidate],
        ohlcv: pd.DataFrame,
    ) -> dict[str, float]:
        """Per-ticker annualized vol from ohlcv['vol_cc_20d'] latest row."""
        default_vol = 0.20  # NASDAQ baseline
        if (
            ohlcv is None
            or ohlcv.empty
            or "vol_cc_20d" not in ohlcv.columns
        ):
            return {c.ticker: default_vol for c in selected}
        sorted_df = ohlcv.sort_values("date") if "date" in ohlcv.columns else ohlcv
        latest = (
            sorted_df.groupby("ticker").tail(1)
            .set_index("ticker")["vol_cc_20d"]
        )
        result: dict[str, float] = {}
        for c in selected:
            vol = latest.get(c.ticker, None)
            if vol is None or not pd.notna(vol) or vol <= 0:
                vol = default_vol
            result[c.ticker] = float(max(vol, 0.05))
        return result
