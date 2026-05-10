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

Edge layer enrichment (when calibration_table.json publish됨):
  - net_edge_5d / edge_tier 채워짐
  - SignalGenerator 의 entry 결정은 여전히 canonical
  - AllocationEngine 은 net_edge 기반 sizing override 활성

활성 wiring (2026-05-10 V3.3 전체 활성화 commit 09cc8f2 / ee3ed4c):
  - Phase 3 exit policies (exit_thesis / partial_exit / signal_decay)
    → position-level BookActions
  - Phase 4 capital expansion (allocation / pyramid / rotation)
    → ADD_TO_WINNER / ROTATE / weight override

BacktestEngine + LivePipeline 모두 BookOptimizer wiring 완료
(commit 5ee271f / 09cc8f2 / 4ced695). 단위 테스트는
test_book_optimizer.py 에 standalone, 통합은 test_backtest_v33_integration.py
+ test_live_monitor_v33.py 참조.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from v3.config.schema import FeatureFlagsConfig
from v3.execution.execution_quality import ExecutionQualityMonitor
from v3.rules.entry import OperationalState
from v3.strategy.allocation import AllocationEngine
from v3.strategy.diagnostics import (
    NoTradeReasonLogger,
    RejectReason,
    TransferCoefficientMonitor,
)
from v3.strategy.edge_calibrator import EdgeCalibrator
from v3.strategy.edge_engine import EdgeEngine
from v3.strategy.edge_tier import EdgeTierSystem
from v3.strategy.exit_thesis import ExitThesisEngine
from v3.strategy.partial_exit import PartialExitEngine
from v3.strategy.pyramid import PyramidPolicyEngine
from v3.strategy.regime_v2 import Regime
from v3.strategy.rotation import CapitalRotationEngine
from v3.strategy.signal import SignalGenerator, TradeSignal
from v3.strategy.signal_decay import SignalDecayEngine
from v3.strategy.types import (
    BookAction,
    DecisionContext,
    EdgeCandidate,
    PositionState,
)


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
        # Phase 3 — exit policies
        exit_thesis: Optional[ExitThesisEngine] = None,
        partial_exit: Optional[PartialExitEngine] = None,
        signal_decay: Optional[SignalDecayEngine] = None,
        # Phase 4 — capital expansion
        allocation: Optional[AllocationEngine] = None,
        pyramid: Optional[PyramidPolicyEngine] = None,
        rotation: Optional[CapitalRotationEngine] = None,
        # Diagnostics (read-only)
        no_trade_logger: Optional[NoTradeReasonLogger] = None,
        tc_monitor: Optional[TransferCoefficientMonitor] = None,
        execution_quality: Optional[ExecutionQualityMonitor] = None,
    ):
        self.signal_gen = signal_gen
        self.features = features
        # Edge layer
        self.edge_calibrator = edge_calibrator
        self.edge_engine = edge_engine
        self.edge_tier = edge_tier
        # Phase 3
        self.exit_thesis = exit_thesis
        self.partial_exit = partial_exit
        self.signal_decay = signal_decay
        # Phase 4
        self.allocation = allocation
        self.pyramid = pyramid
        self.rotation = rotation
        # Diagnostics
        self.no_trade_logger = no_trade_logger
        self.tc_monitor = tc_monitor
        self.execution_quality = execution_quality

    # ── Public API ────────────────────────────────────────────
    def decide(self, *args, **kwargs) -> list[BookAction]:
        """Thin wrapper around decide_with_context() returning only actions.

        Preserved for PR-2.4 callers. New code should prefer
        decide_with_context() to access TradeSignal + EdgeCandidate metadata.
        """
        return self.decide_with_context(*args, **kwargs).actions

    def decide_with_context(
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
        # PR-4.4: Position-level inputs
        positions: Optional[list[PositionState]] = None,
        exit_triggers: Optional[dict[str, str]] = None,
        signal_age_hours: float = 0.0,
        rotations_this_month: int = 0,
        adds_per_position: Optional[dict[str, int]] = None,
        initial_weights: Optional[dict[str, float]] = None,
    ) -> DecisionContext:
        """Generate the day's BookActions.

        Args:
            ohlcv, vol_scores, regime, cost, state: same as SignalGenerator
            as_of: timestamp for action.date (default = today)
            ticker_volume_map, sector_map: per-ticker context maps
            ticker_vol_map: per-ticker annualized vol for slippage_buffer
            liquidity_bucket_map: per-ticker {high/mid/low}
            positions: existing PositionState list for exit/pyramid/rotation
            exit_triggers: {ticker: trigger_name from ExitRules}
            signal_age_hours: shared age for stale-signal refresh logic
            rotations_this_month: count for rotation cap
            adds_per_position: {ticker: count of pyramid adds so far}
            initial_weights: {ticker: original entry weight} for pyramid sizing

        Returns:
            Action-priority sorted list of BookActions.
        """
        as_of = as_of if as_of is not None else pd.Timestamp.now().normalize()
        ticker_vol_map = ticker_vol_map or {}
        liquidity_bucket_map = liquidity_bucket_map or {}
        sector_map = sector_map or {}
        positions = positions or []
        exit_triggers = exit_triggers or {}
        adds_per_position = adds_per_position or {}
        initial_weights = initial_weights or {}

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

        # Step 4: Position-level decisions (Phase 3 — exit_thesis, partial_exit)
        position_actions, keep_positions = self._evaluate_positions(
            positions=positions,
            candidates=candidates,
            exit_triggers=exit_triggers,
            signal_age_hours=signal_age_hours,
            as_of=as_of,
        )

        # Step 5: Pyramid (KEEP positions only) — Phase 4
        pyramid_actions = self._evaluate_pyramid(
            keep_positions=keep_positions,
            adds_per_position=adds_per_position,
            initial_weights=initial_weights,
            as_of=as_of,
        )

        # Step 6: Rotation (KEEP positions × new candidates) — Phase 4
        rotation_actions = self._evaluate_rotation(
            keep_positions=keep_positions,
            candidates=candidates,
            rotations_this_month=rotations_this_month,
            as_of=as_of,
        )

        # Step 7: New entries — AllocationEngine override or SignalGenerator
        entry_actions = self._generate_entries(
            signal=signal,
            candidates=candidates,
            regime=regime,
            sector_map=sector_map,
            existing_tickers={p.ticker for p in positions},
            as_of=as_of,
        )

        actions = position_actions + pyramid_actions + rotation_actions + entry_actions

        # Step 8: Diagnostics (read-only, flag-gated)
        self._record_diagnostics(signal, candidates, actions, as_of)

        # Step 9: Sort by priority
        sorted_actions = sorted(
            actions, key=lambda a: ACTION_PRIORITY.get(a.action_type, 99)
        )
        return DecisionContext(
            actions=sorted_actions,
            signal=signal,
            candidates=candidates,
        )

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

    # ── Internal: position-level evaluation (Phase 3) ─────────
    def _evaluate_positions(
        self,
        positions: list[PositionState],
        candidates: list[EdgeCandidate],
        exit_triggers: dict[str, str],
        signal_age_hours: float,
        as_of: pd.Timestamp,
    ) -> tuple[list[BookAction], list[tuple[PositionState, Optional[EdgeCandidate]]]]:
        """Phase 3 exit policies for each position.

        Returns:
            (position_actions, keep_positions) where keep_positions is a list
            of (PositionState, EdgeCandidate) for KEEP-action positions
            (used by pyramid/rotation downstream).
        """
        actions: list[BookAction] = []
        keep_positions: list[tuple[PositionState, Optional[EdgeCandidate]]] = []

        for position in positions:
            position_edge = self._find_candidate(candidates, position.ticker)
            trigger = exit_triggers.get(position.ticker, "max_hold")

            if self.features.exit_thesis and self.exit_thesis is not None:
                action = self.exit_thesis.decide(
                    position=position,
                    exit_trigger=trigger,
                    current_edge=position_edge,
                    replacement_candidates=[
                        c for c in candidates if c.ticker != position.ticker
                    ],
                    signal_age_hours=signal_age_hours,
                    now=as_of,
                )
            elif self.features.partial_exit and self.partial_exit is not None:
                residual = (
                    position_edge.net_edge_5d
                    if position_edge and position_edge.net_edge_5d is not None
                    else position.residual_edge
                )
                thesis_break = (
                    "vol_contraction" if trigger == "vol_contraction" else None
                )
                action = self.partial_exit.evaluate(
                    position=position,
                    residual_edge=residual,
                    thesis_break=thesis_break,
                    now=as_of,
                )
            else:
                # No exit policy active → KEEP (V3.2.1 ExitRules handled externally)
                action = BookAction(
                    date=as_of,
                    action_type="KEEP",
                    ticker=position.ticker,
                    target_weight=position.current_weight,
                    current_weight=position.current_weight,
                    reason="passthrough_keep",
                    source_policy="passthrough",
                )

            actions.append(action)
            if action.action_type == "KEEP":
                keep_positions.append((position, position_edge))

        return actions, keep_positions

    # ── Internal: pyramid (Phase 4) ───────────────────────────
    def _evaluate_pyramid(
        self,
        keep_positions: list[tuple[PositionState, Optional[EdgeCandidate]]],
        adds_per_position: dict[str, int],
        initial_weights: dict[str, float],
        as_of: pd.Timestamp,
    ) -> list[BookAction]:
        if not (self.features.pyramid and self.pyramid is not None):
            return []

        actions: list[BookAction] = []
        for position, position_edge in keep_positions:
            if position_edge is None:
                continue
            adds = adds_per_position.get(position.ticker, 0)
            initial = initial_weights.get(position.ticker, position.current_weight)
            add_action = self.pyramid.evaluate(
                position=position,
                candidate=position_edge,
                adds_so_far=adds,
                initial_weight=initial,
                now=as_of,
            )
            if add_action is not None:
                actions.append(add_action)
        return actions

    # ── Internal: rotation (Phase 4) ──────────────────────────
    def _evaluate_rotation(
        self,
        keep_positions: list[tuple[PositionState, Optional[EdgeCandidate]]],
        candidates: list[EdgeCandidate],
        rotations_this_month: int,
        as_of: pd.Timestamp,
    ) -> list[BookAction]:
        if not (self.features.rotation and self.rotation is not None):
            return []

        # Candidates not currently held with positive net_edge
        held_tickers = {p.ticker for p, _ in keep_positions}
        new_candidates = [
            c for c in candidates
            if c.ticker not in held_tickers
            and c.entry_pass
            and c.net_edge_5d is not None and c.net_edge_5d > 0
        ]
        # Sort highest first
        new_candidates.sort(key=lambda c: c.net_edge_5d or 0.0, reverse=True)

        actions: list[BookAction] = []
        rotations_used = rotations_this_month
        for position, _ in keep_positions:
            for cand in new_candidates:
                rot = self.rotation.evaluate(
                    position=position,
                    candidate=cand,
                    rotations_this_month=rotations_used,
                    now=as_of,
                )
                if rot is not None:
                    actions.append(rot)
                    rotations_used += 1
                    break  # one rotation per position
        return actions

    # ── Internal: new entries (Phase 4 Allocation or passthrough) ──
    def _generate_entries(
        self,
        signal: TradeSignal,
        candidates: list[EdgeCandidate],
        regime: Regime,
        sector_map: dict[str, str],
        existing_tickers: set[str],
        as_of: pd.Timestamp,
    ) -> list[BookAction]:
        """New entries via AllocationEngine override or SignalGenerator passthrough."""
        if self.features.allocation and self.allocation is not None:
            new_weights = self.allocation.allocate(
                candidates=candidates,
                regime=regime,
                sector_map=sector_map,
            )
            return [
                BookAction(
                    date=as_of,
                    action_type="ADD_NEW",
                    ticker=ticker,
                    target_weight=weight,
                    current_weight=0.0,
                    reason=f"allocation_engine_entry(weight={weight:.4f})",
                    source_policy="AllocationEngine",
                    expected_impact=self._candidate_edge(candidates, ticker),
                )
                for ticker, weight in new_weights.items()
                if ticker not in existing_tickers
            ]

        # Passthrough (V3.2.1 parity)
        if signal.action == "CASH":
            return [BookAction(
                date=as_of,
                action_type="NO_ACTION",
                ticker="",
                target_weight=0.0,
                current_weight=0.0,
                reason=f"CASH (regime={signal.regime_name})",
                source_policy="SignalGenerator",
            )]

        return [
            BookAction(
                date=as_of,
                action_type="ADD_NEW",
                ticker=str(pos["ticker"]),
                target_weight=float(pos.get("weight", 0.0)),
                current_weight=0.0,
                reason=f"signal_generator_entry (opp={pos.get('opportunity', 0.0):.5f})",
                source_policy="SignalGenerator",
                expected_impact=float(pos.get("opportunity", 0.0)),
            )
            for pos in signal.positions
            if str(pos["ticker"]) not in existing_tickers
        ]

    @staticmethod
    def _find_candidate(
        candidates: list[EdgeCandidate], ticker: str,
    ) -> Optional[EdgeCandidate]:
        for c in candidates:
            if c.ticker == ticker:
                return c
        return None

    @staticmethod
    def _candidate_edge(
        candidates: list[EdgeCandidate], ticker: str,
    ) -> Optional[float]:
        c = BookOptimizer._find_candidate(candidates, ticker)
        return c.net_edge_5d if c else None

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
