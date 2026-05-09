"""V3.3 PyramidPolicyEngine — winner-only add-on (PR-4.2).

페르소나 §2 "크게" 직접 처방.
맞은 포지션에는 추가 배팅, 손실 포지션에는 절대 averaging-down 금지.

핵심 invariant (회귀 보호):
  손실 포지션 (pnl ≤ 0)은 어떤 경우에도 add-on 발생하면 안 됨.

Trigger conditions (모두 만족 필요):
  · forbid_averaging_down: pnl_pct > 0 (절대 invariant)
  · min_unrealized_pnl: 1.5%+ 수익 (winner)
  · min_residual_edge: 0.40%+ 잔여 edge
  · min_current_tier: A 이상 (S/A만)
  · max_adds_per_position 미초과
  · vol_expansion still active (선택)

Add weight formula:
  new_weight = min(
      current_weight + initial_weight × add_size_fraction (0.50),
      max_single_weight_after_add (0.40)
  )

ExitThesisEngine과의 관계:
  ExitThesis 결과가 KEEP인 winner에 대해 PyramidEngine.evaluate() 호출.
  EXIT/TRIM/ROTATE 후보에는 add-on 평가 안 함.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import BookAction, EdgeCandidate, PositionState


# Tier rank — pyramid evaluation에서 사용
_TIER_RANK: dict[str, int] = {
    "BLOCKED": -1, "C": 0, "B": 1, "A": 2, "S": 3,
}


def _tier_rank(tier: Optional[str]) -> int:
    if tier is None:
        return -1
    return _TIER_RANK.get(tier, -1)


# ──────────────────────────────────────────────────────────────
# PyramidPolicy
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PyramidPolicy:
    """Pyramid add-on policy parameters.

    add_only_to_winners: True (semantic flag — explicit)
    forbid_averaging_down: True (절대 invariant, 변경 금지)
    min_unrealized_pnl_for_add: 1.5% — winner 검증 임계
    min_residual_edge_for_add: 0.40% — alpha 살아있음
    min_current_tier_for_add: "A" — 저신뢰 신호로 add-on 금지
    max_adds_per_position: 2 — 최대 2회 add (총 3 trance)
    add_size_fraction_of_initial: 0.50 — 매 add 시 initial의 50% 추가
    max_single_weight_after_add: 0.40 — add 후도 단일 종목 40% cap
    """
    enabled: bool = True
    add_only_to_winners: bool = True
    forbid_averaging_down: bool = True
    min_unrealized_pnl_for_add: float = 0.015
    min_residual_edge_for_add: float = 0.004
    min_current_tier_for_add: str = "A"
    max_adds_per_position: int = 2
    add_size_fraction_of_initial: float = 0.50
    max_single_weight_after_add: float = 0.40


# ──────────────────────────────────────────────────────────────
# PyramidPolicyEngine
# ──────────────────────────────────────────────────────────────
class PyramidPolicyEngine:
    """Winner-only add-on engine.

    Implements DecisionPolicy protocol.
    evaluate() returns ADD_TO_WINNER BookAction or None.

    Use:
        eng = PyramidPolicyEngine(policy=PyramidPolicy())
        action = eng.evaluate(position, candidate, adds_so_far=0, initial_weight=0.20)
        if action is not None:
            # apply add-on
    """

    name = "pyramid"

    def __init__(self, policy: PyramidPolicy):
        self.policy = policy
        self._enabled = policy.enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        position: PositionState,
        candidate: EdgeCandidate,
        adds_so_far: int,
        initial_weight: float,
        now: Optional[pd.Timestamp] = None,
    ) -> Optional[BookAction]:
        """Evaluate add-on opportunity for this winner.

        Args:
            position: PositionState (must have pnl_pct, current_weight)
            candidate: fresh EdgeCandidate for this position's ticker
            adds_so_far: number of add-ons already applied to this position
            initial_weight: original entry weight (for sizing add)
            now: timestamp for action.date

        Returns:
            ADD_TO_WINNER BookAction if all gates pass, else None.
        """
        if not self.policy.enabled:
            return None

        # Gate 1: max adds limit
        if adds_so_far >= self.policy.max_adds_per_position:
            return None

        # Gate 2: 절대 invariant — 손실 포지션 add-on 금지 (averaging-down 차단)
        if self.policy.forbid_averaging_down and position.pnl_pct <= 0:
            return None

        # Gate 3: minimum unrealized profit
        if position.pnl_pct < self.policy.min_unrealized_pnl_for_add:
            return None

        # Gate 4: residual edge still strong
        net_edge = candidate.net_edge_5d
        if net_edge is None or net_edge < self.policy.min_residual_edge_for_add:
            return None

        # Gate 5: tier requirement (A or S)
        if _tier_rank(candidate.edge_tier) < _tier_rank(
            self.policy.min_current_tier_for_add
        ):
            return None

        # All gates passed — compute add weight
        add_amount = initial_weight * self.policy.add_size_fraction_of_initial
        new_weight = min(
            position.current_weight + add_amount,
            self.policy.max_single_weight_after_add,
        )

        # Sanity: new_weight > current_weight (else nothing to add)
        if new_weight <= position.current_weight + 1e-9:
            return None

        now = now if now is not None else pd.Timestamp.now()
        return BookAction(
            date=now,
            action_type="ADD_TO_WINNER",
            ticker=position.ticker,
            target_weight=new_weight,
            current_weight=position.current_weight,
            reason=(
                f"winner_add_on(pnl={position.pnl_pct:.4f}, "
                f"residual={net_edge:.5f}, tier={candidate.edge_tier}, "
                f"adds_so_far={adds_so_far})"
            ),
            source_policy=self.name,
            expected_impact=net_edge,
        )
