"""V3.3 BookOptimizer — daily decision engine wrapping SignalGenerator + Edge layer.

Pipeline (PR-2.4 scope):
  V3.2.1 SignalGenerator (unchanged)
       ↓ TradeSignal (positions, opportunity_map, rejections)
       ↓
  BookOptimizer.decide()
       ├─ Convert opportunities → EdgeCandidate list
       ├─ (flag) EdgeCalibrator → expected_return_5d
       ├─ (flag) EdgeEngine → net_edge_5d
       ├─ (flag) EdgeTierSystem → edge_tier
       ├─ TradeSignal.positions → ADD_NEW BookAction (passthrough sizing)
       ├─ Rejections → NoTradeLog (DiagnosticSink)
       └─ TC Monitor snapshot
       ↓ list[BookAction]

Parity guarantee (V3.3_DESIGN §11):
  features.all_off() → SignalGenerator positions become ADD_NEW BookActions
  1:1, no decisions changed, no Edge fields populated. Diagnostics also OFF.

Edge feature flags in PR-2.4 are enrichment-only:
  - net_edge_5d / edge_tier populated for observability
  - SignalGenerator's entry decisions remain canonical
  - AllocationEngine sizing override deferred to PR-4.1

Wiring extensions (later PRs):
  - PR-3.x exit policies → modify position-level actions
  - PR-4.1 AllocationEngine → override sizing based on net_edge
  - PR-4.2 Pyramid → ADD_TO_WINNER actions
  - PR-4.3 Rotation → ROTATE actions

Backtest engine + live pipeline integration deferred (PR-2.5 or later).
This module is standalone-tested via injected SignalGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from v3.config.schema import FeatureFlagsConfig
from v3.execution.execution_quality import ExecutionQualityMonitor
from v3.rules.entry import OperationalState
from v3.strategy.diagnostics import (
    NoTradeReasonLogger,
    RejectReason,
    TransferCoefficientMonitor,
)
from v3.strategy.edge_calibrator import EdgeCalibrator
from v3.strategy.edge_engine import EdgeEngine
from v3.strategy.edge_tier import EdgeTierSystem
from v3.strategy.regime_v2 import Regime
from v3.strategy.signal import SignalGenerator, TradeSignal
from v3.strategy.types import BookAction, EdgeCandidate


# ──────────────────────────────────────────────────────────────
# Action priority (lower number = higher priority)
# ──────────────────────────────────────────────────────────────
ACTION_PRIORITY: dict[str, int] = {
    "EXIT": 0,           # risk-based exits first
    "BLOCKED": 1,
    "ROTATE": 2,
    "ADD_TO_WINNER": 3,  # pyramid before new entry
    "ADD_NEW": 4,
    "TRIM": 5,
    "KEEP": 6,
    "NO_ACTION": 7,
}


# ──────────────────────────────────────────────────────────────
# BookOptimizer
# ──────────────────────────────────────────────────────────────
class BookOptimizer:
    """V3.3 daily decision engine.

    Wraps V3.2.1 SignalGenerator and adds:
      - Edge layer enrichment (flag-gated)
      - Diagnostic sinks (NoTrade, TC, ExecutionQuality)
      - BookAction normalization (uniform output for backtest + live)

    PR-2.4 scope: enrichment + diagnostics. Decision logic unchanged from
    V3.2.1. Phase 3+ PRs add exit/rotation/pyramid policies.

    Usage:
        bo = BookOptimizer(
            signal_gen=SignalGenerator(...),
            edge_calibrator=EdgeCalibrator.from_file(...),
            edge_engine=EdgeEngine(),
            edge_tier=EdgeTierSystem(...),
            no_trade_logger=NoTradeReasonLogger(),
            tc_monitor=TransferCoefficientMonitor(),
            execution_quality=ExecutionQualityMonitor(),
            features=FeatureFlagsConfig(),
        )
        actions = bo.decide(ohlcv, vol_scores, regime, cost, state, ...)
    """

    def __init__(
        self,
        signal_gen: SignalGenerator,
        features: FeatureFlagsConfig,
        edge_calibrator: Optional[EdgeCalibrator] = None,
        edge_engine: Optional[EdgeEngine] = None,
        edge_tier: Optional[EdgeTierSystem] = None,
        no_trade_logger: Optional[NoTradeReasonLogger] = None,
        tc_monitor: Optional[TransferCoefficientMonitor] = None,
        execution_quality: Optional[ExecutionQualityMonitor] = None,
    ):
        self.signal_gen = signal_gen
        self.features = features
        self.edge_calibrator = edge_calibrator
        self.edge_engine = edge_engine
        self.edge_tier = edge_tier
        self.no_trade_logger = no_trade_logger
        self.tc_monitor = tc_monitor
        self.execution_quality = execution_quality

    # ── Public API ────────────────────────────────────────────
    def decide(
        self,
        ohlcv: pd.DataFrame,
        vol_scores: pd.DataFrame,
        regime: Regime,
        cost: float,
        state: OperationalState,
        as_of: Optional[pd.Timestamp] = None,
        ticker_volume_map: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
        ticker_vol_map: Optional[dict[str, float]] = None,
        liquidity_bucket_map: Optional[dict[str, str]] = None,
    ) -> list[BookAction]:
        """Generate the day's BookActions.

        Args:
            ohlcv, vol_scores, regime, cost, state: same as SignalGenerator
            as_of: timestamp for action.date (default = today)
            ticker_volume_map, sector_map: per-ticker context maps
            ticker_vol_map: per-ticker annualized vol for slippage_buffer
            liquidity_bucket_map: per-ticker {high/mid/low}

        Returns:
            Action-priority sorted list of BookActions.
        """
        as_of = as_of if as_of is not None else pd.Timestamp.now().normalize()
        ticker_vol_map = ticker_vol_map or {}
        liquidity_bucket_map = liquidity_bucket_map or {}
        sector_map = sector_map or {}

        # Step 1: V3.2.1 SignalGenerator (unchanged)
        signal = self.signal_gen.generate(
            ohlcv=ohlcv,
            vol_scores=vol_scores,
            regime=regime,
            cost=cost,
            state=state,
            ticker_volume_map=ticker_volume_map,
            sector_map=sector_map,
        )

        # Step 2: Build EdgeCandidate list from opportunity_map
        candidates = self._build_candidates(
            signal=signal,
            regime=regime,
            as_of=as_of,
            sector_map=sector_map,
            liquidity_bucket_map=liquidity_bucket_map,
        )

        # Step 3: (flag) Edge layer enrichment
        candidates = self._apply_edge_layer(candidates, ticker_vol_map)

        # Step 4: Generate BookActions from SignalGenerator positions
        actions = self._signal_to_actions(signal, as_of)

        # Step 5: Diagnostics (read-only, flag-gated)
        self._record_diagnostics(signal, candidates, actions, as_of)

        # Step 6: Sort by priority
        return sorted(actions, key=lambda a: ACTION_PRIORITY.get(a.action_type, 99))

    # ── Internal: candidate construction ──────────────────────
    def _build_candidates(
        self,
        signal: TradeSignal,
        regime: Regime,
        as_of: pd.Timestamp,
        sector_map: dict[str, str],
        liquidity_bucket_map: dict[str, str],
    ) -> list[EdgeCandidate]:
        """Convert TradeSignal.opportunity_map into EdgeCandidate objects.

        Direction/conviction are reconstructed from approved positions
        (matched by ticker). For non-approved candidates (in opportunity_map
        but not selected), direction/conviction default to 0/0.
        """
        position_lookup = {
            p["ticker"]: p for p in signal.positions
        }

        candidates: list[EdgeCandidate] = []
        for ticker, opp in signal.opportunity_map.items():
            pos = position_lookup.get(ticker)
            if pos is not None:
                direction = float(pos.get("direction", 0.0))
                conviction = float(pos.get("conviction", 0.0))
            else:
                direction = 0.0
                conviction = 0.0

            candidates.append(EdgeCandidate(
                date=as_of,
                ticker=ticker,
                direction=direction,
                conviction=conviction,
                raw_opportunity=float(opp),
                regime=regime.name,
                regime_score=regime.score,
                sector=sector_map.get(ticker),
                liquidity_bucket=liquidity_bucket_map.get(ticker),
            ))

        return candidates

    # ── Internal: Edge layer (flag-gated) ─────────────────────
    def _apply_edge_layer(
        self,
        candidates: list[EdgeCandidate],
        ticker_vol_map: dict[str, float],
    ) -> list[EdgeCandidate]:
        """Apply EdgeCalibrator → EdgeEngine → EdgeTierSystem (each flag-gated)."""
        if not candidates:
            return candidates

        if self.features.edge_calibrator and self.edge_calibrator is not None:
            candidates = self.edge_calibrator.calibrate_batch(candidates)

        if self.features.edge_engine and self.edge_engine is not None:
            candidates = self.edge_engine.compute_batch(
                candidates, ticker_vols=ticker_vol_map,
            )

        if self.features.edge_tier and self.edge_tier is not None:
            candidates = self.edge_tier.assign_batch(candidates)

        return candidates

    # ── Internal: signal → actions (parity passthrough) ───────
    def _signal_to_actions(
        self,
        signal: TradeSignal,
        as_of: pd.Timestamp,
    ) -> list[BookAction]:
        """Convert TradeSignal positions → ADD_NEW BookActions.

        Parity contract: 1:1 mapping. SignalGenerator decisions canonical.
        """
        actions: list[BookAction] = []

        if signal.action == "CASH":
            actions.append(BookAction(
                date=as_of,
                action_type="NO_ACTION",
                ticker="",
                target_weight=0.0,
                current_weight=0.0,
                reason=f"CASH (regime={signal.regime_name})",
                source_policy="SignalGenerator",
            ))
            return actions

        for pos in signal.positions:
            actions.append(BookAction(
                date=as_of,
                action_type="ADD_NEW",
                ticker=str(pos["ticker"]),
                target_weight=float(pos.get("weight", 0.0)),
                current_weight=0.0,
                reason=f"signal_generator_entry (opp={pos.get('opportunity', 0.0):.5f})",
                source_policy="SignalGenerator",
                expected_impact=float(pos.get("opportunity", 0.0)),
            ))
        return actions

    # ── Internal: diagnostics ─────────────────────────────────
    def _record_diagnostics(
        self,
        signal: TradeSignal,
        candidates: list[EdgeCandidate],
        actions: list[BookAction],
        as_of: pd.Timestamp,
    ) -> None:
        """Read-only diagnostic sinks (each flag-gated)."""
        # NoTradeLogger: rejected candidates
        if self.features.no_trade_logger and self.no_trade_logger is not None:
            entered_tickers = {a.ticker for a in actions if a.action_type == "ADD_NEW"}
            for c in candidates:
                if c.ticker in entered_tickers:
                    continue
                # Reject reason: check explicit reject_reason from Edge layer first
                reason = c.reject_reason or self._infer_reject_reason(c, signal)
                self.no_trade_logger.record({
                    "date": as_of,
                    "ticker": c.ticker,
                    "stage": self._infer_stage(c),
                    "reject_reason": reason,
                    "raw_opportunity": c.raw_opportunity,
                    "regime": c.regime,
                    "net_edge_5d": c.net_edge_5d,
                    "edge_tier": c.edge_tier,
                })

        # TC Monitor: snapshot
        if self.features.tc_monitor and self.tc_monitor is not None:
            final_weights = {
                a.ticker: a.target_weight
                for a in actions if a.action_type == "ADD_NEW"
            }
            snap = self.tc_monitor.compute(candidates, final_weights, as_of=as_of)
            self.tc_monitor.record({"snapshot": snap})

    @staticmethod
    def _infer_stage(c: EdgeCandidate) -> str:
        if c.net_edge_5d is not None:
            return "edge"
        if c.expected_return_5d is not None:
            return "calibration"
        return "opportunity"

    @staticmethod
    def _infer_reject_reason(c: EdgeCandidate, signal: TradeSignal) -> str:
        """Best-effort reason inference for rejected candidates without explicit tag."""
        if c.raw_opportunity <= signal.opportunity_gate:
            return RejectReason.RAW_OPPORTUNITY_TOO_LOW.value
        if c.edge_tier in ("C", "BLOCKED"):
            return RejectReason.TIER_BELOW_MIN.value
        # Otherwise: passed opportunity gate but rejected operationally
        # Common cases: position_limit, monthly_cap, sector_cap.
        # SignalGenerator.rejection_reasons aggregates these but not per-ticker.
        return RejectReason.POSITION_LIMIT.value

    # ── Lifecycle hooks (for future BookOptimizer integration) ──
    def flush_diagnostics(self) -> None:
        """Persist all buffered diagnostic sinks."""
        if self.no_trade_logger is not None:
            self.no_trade_logger.flush()
        if self.tc_monitor is not None:
            self.tc_monitor.flush()
        if self.execution_quality is not None:
            self.execution_quality.flush()
