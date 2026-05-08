"""V3.3 PartialExitEngine — 부분 청산 / 부분 유지 (PR-3.4).

전량 청산만으로 대응하지 않고, residual edge + drawdown_from_peak +
vol_contraction 상태에 따라 부분 청산/부분 유지/전량 청산 분리.

의사결정 테이블 (priority 순):
  1. vol_contraction confirmed → EXIT (전량)
  2. residual_edge ≤ exit_threshold → EXIT (전량)
  3. profit_protection (max +3% 후 -1.5% 하락 + edge 약화) → TRIM 50%
  4. edge_weakening (residual < trim_threshold) → TRIM 30%
  5. else → KEEP

페르소나 정합성 (Phase 25.2 sizer floor):
  min_remaining_weight = 0.15.
  TRIM 후 잔여가 floor 미만이면 EXIT 전환.

ExitThesisEngine과의 관계:
  ExitThesisEngine은 단순 trim_fraction (default 0.30) 사용.
  PartialExitEngine은 더 정교한 의사결정 (profit protection 등) 추가.
  통합 시 PartialExitEngine 우선 (ExitThesis가 PartialExit 호출).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import BookAction, PositionState


# ──────────────────────────────────────────────────────────────
# PartialExitPolicy
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PartialExitPolicy:
    """Partial exit decision parameters.

    trim_fraction_default: 기본 trim 비율 (50% trim, 50% keep)
    min_remaining_weight: 잔여 비중 하한 (Phase 25.2 sizer floor 0.15)
    profit_protection_trigger: max_unrealized 도달 임계 (+3.0%)
    drawdown_from_peak_trigger: 고점 대비 -1.5% 하락
    trim_if_residual_edge_below: 약화 trim 임계 (0.002)
    exit_if_residual_edge_below: 전량 exit 임계 (0.000)
    full_exit_on_vol_contraction: vol contraction은 부분이 아닌 전량
    weak_edge_trim_fraction: edge 약화 시 trim 비율 (30%)
    """
    trim_fraction_default: float = 0.50
    min_remaining_weight: float = 0.15
    profit_protection_trigger: float = 0.030
    drawdown_from_peak_trigger: float = -0.015
    trim_if_residual_edge_below: float = 0.002
    exit_if_residual_edge_below: float = 0.000
    full_exit_on_vol_contraction: bool = True
    weak_edge_trim_fraction: float = 0.30


# ──────────────────────────────────────────────────────────────
# PartialExitEngine
# ──────────────────────────────────────────────────────────────
class PartialExitEngine:
    """Partial exit decision based on residual_edge + drawdown_from_peak.

    Implements DecisionPolicy protocol.

    Use:
        eng = PartialExitEngine(policy=PartialExitPolicy())
        action = eng.evaluate(
            position=ps,
            residual_edge=0.001,
            thesis_break="vol_contraction",  # or None
        )
    """

    name = "partial_exit"

    def __init__(self, policy: PartialExitPolicy, enabled: bool = True):
        self.policy = policy
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Public API ────────────────────────────────────────────
    def evaluate(
        self,
        position: PositionState,
        residual_edge: float,
        thesis_break: Optional[str] = None,
        now: Optional[pd.Timestamp] = None,
    ) -> BookAction:
        """Partial exit decision.

        Args:
            position: PositionState snapshot
            residual_edge: current effective edge (post-decay)
            thesis_break: optional reason ("vol_contraction" forces full exit)
            now: timestamp for action.date

        Returns:
            BookAction (EXIT / TRIM / KEEP).
        """
        now = now if now is not None else pd.Timestamp.now()

        # Step 1: vol_contraction → full exit
        if (
            self.policy.full_exit_on_vol_contraction
            and thesis_break == "vol_contraction"
        ):
            return self._exit_full(position, now, reason="vol_contraction_confirmed")

        # Step 2: residual_edge ≤ exit threshold → full exit
        if residual_edge <= self.policy.exit_if_residual_edge_below:
            return self._exit_full(
                position, now,
                reason=f"residual_edge_negative({residual_edge:.5f})",
            )

        # Step 3: profit protection — max +3% 후 -1.5% 하락 + edge 약화
        if (
            position.max_unrealized_pnl_pct >= self.policy.profit_protection_trigger
            and position.drawdown_from_peak_pct <= self.policy.drawdown_from_peak_trigger
            and residual_edge < self.policy.trim_if_residual_edge_below
        ):
            return self._trim_or_exit(
                position, now,
                fraction=self.policy.trim_fraction_default,
                reason=(
                    f"winner_protection_trim(max_pnl={position.max_unrealized_pnl_pct:.3f}, "
                    f"dd={position.drawdown_from_peak_pct:.3f}, "
                    f"residual={residual_edge:.5f})"
                ),
                expected_impact=residual_edge,
            )

        # Step 4: edge weakening → smaller trim
        if residual_edge < self.policy.trim_if_residual_edge_below:
            return self._trim_or_exit(
                position, now,
                fraction=self.policy.weak_edge_trim_fraction,
                reason=f"weak_residual_edge({residual_edge:.5f})",
                expected_impact=residual_edge,
            )

        # Step 5: KEEP
        return BookAction(
            date=now,
            action_type="KEEP",
            ticker=position.ticker,
            target_weight=position.current_weight,
            current_weight=position.current_weight,
            reason=f"thesis_alive(residual={residual_edge:.5f})",
            source_policy=self.name,
            expected_impact=residual_edge,
        )

    # ── Internal helpers ──────────────────────────────────────
    def _trim_or_exit(
        self,
        position: PositionState,
        now: pd.Timestamp,
        fraction: float,
        reason: str,
        expected_impact: Optional[float] = None,
    ) -> BookAction:
        """TRIM by fraction. If remaining < min_remaining_weight, convert to EXIT."""
        new_weight = position.current_weight * (1.0 - fraction)
        if new_weight < self.policy.min_remaining_weight:
            return self._exit_full(
                position, now,
                reason=f"{reason}_below_floor",
                expected_impact=expected_impact,
            )
        return BookAction(
            date=now,
            action_type="TRIM",
            ticker=position.ticker,
            target_weight=new_weight,
            current_weight=position.current_weight,
            reason=reason,
            source_policy=self.name,
            expected_impact=expected_impact,
        )

    def _exit_full(
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
