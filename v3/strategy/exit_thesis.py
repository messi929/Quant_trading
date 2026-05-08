"""V3.3 Exit policy — Conditional Veto + ExitThesisEngine (PR-3.1, PR-3.2).

PR-3.1 (this commit): ExitPolicy + Conditional Veto helpers.
PR-3.2 (next): ExitThesisEngine.decide() 본체 (HOLD/REDUCE/ROTATE/EXIT).

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

import pandas as pd


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
