"""V3.3 CapitalRotationEngine — capital reallocation (PR-4.3).

페르소나 §1 "확신" 보강. 청산 = 자본 재배분 판단 (단순 매도 아님).

ExitThesisEngine과의 차이:
  ExitThesisEngine: 시간 기반 trigger 시 replacement 평가 (단일 의사결정)
  CapitalRotationEngine: 명시적 rotation 정책 (월간 cap, tier rank 명시)

Trigger conditions (모두 만족 필요):
  · enabled
  · rotations_this_month < max_rotations_per_month
  · position.holding_days ≥ min_holding_days_before_rotation
  · candidate.edge_tier ≥ min_candidate_tier (default "A")
  · forbid_lower_tier_replace_higher: candidate ≥ current tier
  · candidate.net_edge - position.residual_edge > switching + rotation_margin

Rotation BookAction:
  action_type="ROTATE", ticker=position.ticker, target_weight=0.0,
  replacement_ticker=candidate.ticker.

신규 진입 weight는 별도 (AllocationEngine이 다음 사이클에서 계산).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import BookAction, EdgeCandidate, PositionState


# Tier rank — module-local
_TIER_RANK: dict[str, int] = {
    "BLOCKED": -1, "C": 0, "B": 1, "A": 2, "S": 3,
}


def _tier_rank(tier: Optional[str]) -> int:
    if tier is None:
        return -1
    return _TIER_RANK.get(tier, -1)


# ──────────────────────────────────────────────────────────────
# RotationPolicy
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RotationPolicy:
    """Rotation decision policy parameters.

    rotation_margin: edge_gap > switching_cost + this → ROTATE
    min_candidate_tier: 후보 최소 tier (default "A")
    allow_s_replace_a: S can replace A (default True)
    forbid_lower_tier_replace_higher: 명시적 enforcement
    max_rotations_per_month: 월간 회전 cap (turnover 제어)
    min_holding_days_before_rotation: 신규 포지션 최소 보유 일수
    """
    enabled: bool = True
    rotation_margin: float = 0.0025
    min_candidate_tier: str = "A"
    allow_s_replace_a: bool = True
    forbid_lower_tier_replace_higher: bool = True
    max_rotations_per_month: int = 4
    min_holding_days_before_rotation: int = 1


# ──────────────────────────────────────────────────────────────
# CapitalRotationEngine
# ──────────────────────────────────────────────────────────────
class CapitalRotationEngine:
    """Replacement decision: 기존 포지션 vs 신규 후보.

    Implements DecisionPolicy protocol.
    evaluate() returns ROTATE BookAction or None.

    Use:
        eng = CapitalRotationEngine(
            policy=RotationPolicy(),
            switching_cost_fn=lambda a, b: 0.001,
        )
        action = eng.evaluate(position, candidate, rotations_this_month=2)
    """

    name = "rotation"

    def __init__(
        self,
        policy: RotationPolicy,
        switching_cost_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.policy = policy
        self._switching_cost = switching_cost_fn or (lambda a, b: 0.001)
        self._enabled = policy.enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        position: PositionState,
        candidate: EdgeCandidate,
        rotations_this_month: int,
        now: Optional[pd.Timestamp] = None,
    ) -> Optional[BookAction]:
        """Evaluate rotation opportunity.

        Args:
            position: existing PositionState
            candidate: new candidate to potentially replace position
            rotations_this_month: count of rotations already executed this month
            now: timestamp for action.date

        Returns:
            ROTATE BookAction or None.
        """
        if not self.policy.enabled:
            return None

        # Gate 1: monthly rotation cap
        if rotations_this_month >= self.policy.max_rotations_per_month:
            return None

        # Gate 2: minimum holding days
        if position.holding_days < self.policy.min_holding_days_before_rotation:
            return None

        # Gate 3: candidate tier requirement
        if _tier_rank(candidate.edge_tier) < _tier_rank(
            self.policy.min_candidate_tier
        ):
            return None

        # Gate 4: forbid lower tier replacing higher
        if self.policy.forbid_lower_tier_replace_higher:
            if _tier_rank(candidate.edge_tier) < _tier_rank(position.current_tier):
                return None

        # Gate 5: net_edge sanity
        if candidate.net_edge_5d is None:
            return None

        # Edge improvement check
        rotation_edge = candidate.net_edge_5d - position.residual_edge
        switching = self._switching_cost(position.ticker, candidate.ticker)
        hurdle = switching + self.policy.rotation_margin

        if rotation_edge <= hurdle:
            return None

        now = now if now is not None else pd.Timestamp.now()
        return BookAction(
            date=now,
            action_type="ROTATE",
            ticker=position.ticker,
            target_weight=0.0,
            current_weight=position.current_weight,
            reason=(
                f"rotation_edge={rotation_edge:.5f}>"
                f"hurdle={hurdle:.5f}(switch={switching:.5f}+margin={self.policy.rotation_margin:.5f}); "
                f"tier:{position.current_tier}→{candidate.edge_tier}"
            ),
            source_policy=self.name,
            replacement_ticker=candidate.ticker,
            expected_impact=rotation_edge,
        )
