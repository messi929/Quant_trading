"""V3.3 Exit policy — Conditional Veto + ExitThesisEngine (PR-3.1, PR-3.2).

PR-3.1: ExitPolicy + Conditional Veto helpers.
PR-3.2: ExitThesisEngine.decide() 본체 (HOLD/REDUCE/ROTATE/EXIT).

CONDITIONAL VETO 정책 (FOLLOW_UPS 1순위 해결):
  Risk-based exits (portfolio_stop, vol_contraction, dynamic_stop_mae,
  circuit_breaker) → 무조건 청산 (veto 금지).

  Time-based exits (profit_take, max_hold) → residual_edge 재평가 후 결정:
    residual_edge > hold_min_residual_edge: KEEP (vetoed)
    residual_edge ≤ threshold: EXIT

STALE SIGNAL 처리:
  V3.2.1 buggy: 8h staleness threshold < KR↔US 14h gap → always stale
  → unconditional fire (veto 작동 안 함, 4/21 ADI +9.02% 자름)

  V3.3 fix:
    - max_signal_staleness_hours: 16.0 (KR↔US 14h + 여유)
    - refresh_at_session_start: true (세션 시작 시 즉시 generate)

V3.2.1 ExitRules와의 관계:
  ExitRules (v3/rules/exit.py) 그대로 유지. ExitThesisEngine은 ExitRules의
  trigger를 받아서 conditional veto 로직 추가. features.exit_thesis OFF 시
  V3.2.1 동작 (ExitRules.check 그대로) 보존.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import BookAction, EdgeCandidate, PositionState


# ──────────────────────────────────────────────────────────────
# Risk-based exits (절대 veto 금지)
# ──────────────────────────────────────────────────────────────
RISK_BASED_EXITS: frozenset[str] = frozenset({
    "portfolio_stop",
    "dynamic_stop_mae",
    "vol_contraction",
    "circuit_breaker",
})

# Time-based / opportunity-based (재평가 후 veto 가능)
TIME_BASED_EXITS: frozenset[str] = frozenset({
    "profit_take",
    "max_hold",
})


# ──────────────────────────────────────────────────────────────
# ExitPolicy
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExitPolicy:
    """Exit decision policy parameters.

    hold_min_residual_edge: residual edge가 이보다 크면 KEEP (veto)
    reduce_zone_edge: residual edge < reduce_zone_edge 이면 TRIM
    rotation_margin: candidate.net_edge - position.residual_edge >
        switching_cost + rotation_margin 이면 ROTATE
    thesis_break_direction_threshold: direction이 진입 시점 부호와 반대로
        이 값을 초과하면 thesis 깨진 것으로 판단
    max_signal_staleness_hours: signal 이 이보다 오래되면 refresh 필요
        (V3.2.1 8.0 → V3.3 16.0; KR↔US 14h gap + 여유)
    refresh_at_session_start: 세션 시작 시 무조건 generate_signal 호출
    """
    hold_min_residual_edge: float = 0.0
    reduce_zone_edge: float = 0.0015
    rotation_margin: float = 0.0025
    thesis_break_direction_threshold: float = 0.0
    max_signal_staleness_hours: float = 16.0
    refresh_at_session_start: bool = True


# ──────────────────────────────────────────────────────────────
# Helpers — pure functions
# ──────────────────────────────────────────────────────────────
def is_risk_based_exit(trigger: str) -> bool:
    """RISK_BASED_EXITS는 어떤 경우에도 veto되면 안 됨."""
    return trigger in RISK_BASED_EXITS


def is_time_based_exit(trigger: str) -> bool:
    """TIME_BASED는 residual_edge 재평가 후 KEEP 가능."""
    return trigger in TIME_BASED_EXITS


def should_refresh_signal(
    signal_age_hours: float,
    policy: ExitPolicy,
) -> bool:
    """signal이 stale → 세션 시작 시 refresh 필요."""
    return signal_age_hours > policy.max_signal_staleness_hours


def evaluate_conditional_veto(
    exit_trigger: str,
    residual_edge: float,
    policy: ExitPolicy,
) -> tuple[bool, str]:
    """Conditional veto core decision.

    Returns:
        (should_exit, reason)

        should_exit=True  → 청산
        should_exit=False → KEEP (vetoed)

    Risk-based exits: 항상 should_exit=True (veto 금지)
    Time-based exits: residual_edge 양수면 should_exit=False
    그 외 unknown: should_exit=True (안전 우선)
    """
    if is_risk_based_exit(exit_trigger):
        return True, f"risk_exit:{exit_trigger}"

    if is_time_based_exit(exit_trigger):
        if residual_edge > policy.hold_min_residual_edge:
            return False, (
                f"vetoed:{exit_trigger}:"
                f"residual_edge={residual_edge:.5f}"
            )
        return True, (
            f"{exit_trigger}:residual_edge_below_hold_min"
            f"(re={residual_edge:.5f})"
        )

    # Unknown trigger — exit by default (safer)
    return True, f"unknown_trigger:{exit_trigger}"


def signal_age_hours(
    generated_at: pd.Timestamp,
    now: pd.Timestamp | None = None,
) -> float:
    """Signal 생성 시점 ↔ now 시간차 (시간 단위)."""
    now = now if now is not None else pd.Timestamp.now()
    delta = now - generated_at
    return float(delta.total_seconds() / 3600.0)


# ──────────────────────────────────────────────────────────────
# ExitThesisEngine (PR-3.2)
# ──────────────────────────────────────────────────────────────
class ExitThesisEngine:
    """Position-level exit decision: HOLD / REDUCE / ROTATE / EXIT.

    Implements DecisionPolicy protocol. Wraps Conditional Veto + replacement
    + reduce-zone logic.

    Decision tree (priority order):
      1. risk_exit (portfolio_stop, vol_contraction, ...) → EXIT (never veto)
      2. signal stale → refresh (callback)
      3. no fresh signal → EXIT (conservative)
      4. residual_edge ≤ hold_min → EXIT
      5. better replacement (improvement > switching + rotation_margin) → ROTATE
      6. residual_edge < reduce_zone → TRIM
      7. else → KEEP

    Pure function with injected dependencies (signal_refresh_fn, switching_cost_fn).
    """

    name = "exit_thesis"

    def __init__(
        self,
        policy: ExitPolicy,
        signal_refresh_fn: Optional[Callable[[str], Optional[EdgeCandidate]]] = None,
        switching_cost_fn: Optional[Callable[[str, str], float]] = None,
        trim_fraction: float = 0.30,
        enabled: bool = True,
    ):
        """
        Args:
            policy: ExitPolicy thresholds
            signal_refresh_fn: callback returning fresh EdgeCandidate for ticker.
                Returns None if no signal available.
            switching_cost_fn: (current_ticker, replacement_ticker) → cost.
                Default: 0.001 (NASDAQ 왕복).
            trim_fraction: TRIM 시 청산 비율 (default 0.30 = 30% trim, 70% keep)
        """
        self.policy = policy
        self.signal_refresh_fn = signal_refresh_fn
        self.switching_cost_fn = switching_cost_fn or (lambda a, b: 0.001)
        self.trim_fraction = trim_fraction
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Public API ────────────────────────────────────────────
    def decide(
        self,
        position: PositionState,
        exit_trigger: str,
        current_edge: Optional[EdgeCandidate],
        replacement_candidates: list[EdgeCandidate],
        signal_age_hours: float,
        now: Optional[pd.Timestamp] = None,
    ) -> BookAction:
        """Single-position exit decision.

        Args:
            position: PositionState snapshot
            exit_trigger: trigger name from ExitRules (e.g., "profit_take",
                "portfolio_stop")
            current_edge: latest EdgeCandidate for the position's ticker
                (or None if no signal available)
            replacement_candidates: list of fresh EdgeCandidate alternatives
                (with net_edge_5d populated)
            signal_age_hours: how old current_edge is
            now: timestamp for action.date (default = pd.Timestamp.now())

        Returns:
            BookAction (frozen).
        """
        now = now if now is not None else pd.Timestamp.now()

        # Step 1: Risk-based — never veto
        if is_risk_based_exit(exit_trigger):
            return self._exit_action(
                position, now, reason=f"risk_exit:{exit_trigger}",
            )

        # Step 2: Refresh stale signal
        if should_refresh_signal(signal_age_hours, self.policy):
            if self.signal_refresh_fn is not None:
                refreshed = self.signal_refresh_fn(position.ticker)
                if refreshed is not None:
                    current_edge = refreshed

        # Step 3: No fresh signal → conservative exit
        if current_edge is None or current_edge.net_edge_5d is None:
            return self._exit_action(
                position, now, reason="no_fresh_signal",
            )

        residual = current_edge.net_edge_5d

        # Step 4: Residual edge ≤ hold_min → EXIT
        if residual <= self.policy.hold_min_residual_edge:
            return self._exit_action(
                position, now,
                reason=f"residual_edge_below_hold_min({residual:.5f})",
                expected_impact=residual,
            )

        # Step 5: Better replacement → ROTATE
        best = self._best_replacement(replacement_candidates, residual)
        if best is not None:
            improvement, hurdle = self._rotation_metrics(
                position.ticker, best, residual,
            )
            if improvement > hurdle:
                return BookAction(
                    date=now,
                    action_type="ROTATE",
                    ticker=position.ticker,
                    target_weight=0.0,
                    current_weight=position.current_weight,
                    reason=(
                        f"rotation_edge={improvement:.5f}>"
                        f"hurdle={hurdle:.5f}"
                    ),
                    source_policy=self.name,
                    replacement_ticker=best.ticker,
                    expected_impact=improvement,
                )

        # Step 6: Residual weak → TRIM
        if residual < self.policy.reduce_zone_edge:
            new_weight = position.current_weight * (1.0 - self.trim_fraction)
            return BookAction(
                date=now,
                action_type="TRIM",
                ticker=position.ticker,
                target_weight=new_weight,
                current_weight=position.current_weight,
                reason=(
                    f"edge_weakened(residual={residual:.5f}<"
                    f"reduce={self.policy.reduce_zone_edge:.5f})"
                ),
                source_policy=self.name,
                expected_impact=residual,
            )

        # Step 7: KEEP
        return BookAction(
            date=now,
            action_type="KEEP",
            ticker=position.ticker,
            target_weight=position.current_weight,
            current_weight=position.current_weight,
            reason=f"thesis_alive(residual={residual:.5f})",
            source_policy=self.name,
            expected_impact=residual,
        )

    # ── Internal helpers ──────────────────────────────────────
    def _exit_action(
        self,
        position: PositionState,
        now: pd.Timestamp,
        reason: str,
        expected_impact: Optional[float] = None,
    ) -> BookAction:
        return BookAction(
            date=now,
            action_type="EXIT",
            ticker=position.ticker,
            target_weight=0.0,
            current_weight=position.current_weight,
            reason=reason,
            source_policy=self.name,
            expected_impact=expected_impact,
        )

    def _best_replacement(
        self,
        candidates: list[EdgeCandidate],
        current_residual: float,
    ) -> Optional[EdgeCandidate]:
        """Find replacement candidate with strictly higher net_edge."""
        if not candidates:
            return None
        valid = [
            c for c in candidates
            if c.net_edge_5d is not None and c.net_edge_5d > current_residual
        ]
        if not valid:
            return None
        return max(valid, key=lambda c: c.net_edge_5d or 0.0)

    def _rotation_metrics(
        self,
        current_ticker: str,
        replacement: EdgeCandidate,
        current_residual: float,
    ) -> tuple[float, float]:
        """Returns (improvement, hurdle)."""
        switching = self.switching_cost_fn(current_ticker, replacement.ticker)
        improvement = (replacement.net_edge_5d or 0.0) - current_residual
        hurdle = switching + self.policy.rotation_margin
        return improvement, hurdle
