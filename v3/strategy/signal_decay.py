"""V3.3 SignalDecayEngine — alpha별 holding period (PR-3.3).

모든 신호를 동일한 max_hold=5일로 처리하지 않고, dominant_alpha별 expected
holding period + decay half-life을 다르게 적용.

Pipeline position:
  Position holding 중에 ExitRules.check() trigger
       ↓
  ExitThesisEngine.decide() 본체
       ↓
  (선택) SignalDecayEngine.apply_decay(position, current_edge)
       ↓
  decay-adjusted residual_edge → ExitThesisEngine 의사결정

Decay formula:
  decay_multiplier = exp(-holding_days / half_life_days)
  effective_residual = current_net_edge × decay_multiplier

Reset:
  최신 신호가 다시 강화되면 (S/A tier + same direction) decay floor 0.75 적용.

Alpha profiles (default):
  trend     : 10일 holding, 5.0일 half-life, allow_hold_extension=True
  reversion : 3일 holding, 1.5일 half-life, profit_take_fast=True
  vol_expansion_only : 5일 holding, 2.5일 half-life

V3.2.1과의 관계:
  V3.2.1 max_hold_days=5 단일값. 본 모듈 활성화 시 alpha별 차등 적용.
  features.signal_decay=false → V3.2.1 동작 보존.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import EdgeCandidate, PositionState


# ──────────────────────────────────────────────────────────────
# AlphaProfile
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlphaProfile:
    """Per-alpha decay & holding profile.

    expected_holding_days: typical hold horizon for this alpha
    decay_half_life_days: time to half conviction
    profit_take_fast: TP를 빠르게 (V3.2.1 time_decay_targets 가속)
    allow_hold_extension: max_hold 도달해도 strong residual 시 연장 허용
    """
    expected_holding_days: int
    decay_half_life_days: float
    profit_take_fast: bool = False
    allow_hold_extension: bool = False


# ──────────────────────────────────────────────────────────────
# SignalDecayPolicy
# ──────────────────────────────────────────────────────────────
def _default_alpha_profiles() -> dict[str, AlphaProfile]:
    return {
        "trend": AlphaProfile(
            expected_holding_days=10,
            decay_half_life_days=5.0,
            allow_hold_extension=True,
        ),
        "reversion": AlphaProfile(
            expected_holding_days=3,
            decay_half_life_days=1.5,
            profit_take_fast=True,
        ),
        "vol_expansion_only": AlphaProfile(
            expected_holding_days=5,
            decay_half_life_days=2.5,
        ),
    }


@dataclass(frozen=True)
class SignalDecayPolicy:
    """Signal decay configuration.

    default_max_hold_days: fallback for unknown alphas
    default_half_life: fallback for unknown alphas
    alpha_profiles: per-alpha overrides
    reset_min_tier: 최소 tier for decay reset (S or A)
    reset_min_stability: 미사용 (Phase 5+ Stability filter 통합 시 활용)
    require_direction_same_sign_for_reset: entry/current direction 부호 일치 요구
    reset_floor: decay multiplier 하한 (reset 조건 충족 시)
    """
    default_max_hold_days: int = 5
    default_half_life: float = 2.5
    alpha_profiles: dict[str, AlphaProfile] = field(
        default_factory=_default_alpha_profiles
    )
    reset_min_tier: str = "A"
    reset_min_stability: float = 0.75
    require_direction_same_sign_for_reset: bool = True
    reset_floor: float = 0.75


# ──────────────────────────────────────────────────────────────
# SignalDecayEngine
# ──────────────────────────────────────────────────────────────
TIER_RANK_LOCAL: dict[str, int] = {
    "BLOCKED": -1, "C": 0, "B": 1, "A": 2, "S": 3,
}


class SignalDecayEngine:
    """Apply alpha-specific decay to residual_edge.

    Implements DecisionPolicy protocol.

    Methods:
        expected_holding_days(alpha_name) → int
        apply_decay(position, current_value, current_signal=None) → float
        is_reconfirmed(position, signal) → bool
        dominant_alpha(directional_alphas) → str (static helper)
    """

    name = "signal_decay"

    def __init__(self, policy: SignalDecayPolicy, enabled: bool = True):
        self.policy = policy
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Holding period ────────────────────────────────────────
    def expected_holding_days(self, dominant_alpha: str) -> int:
        profile = self.policy.alpha_profiles.get(dominant_alpha)
        if profile is None:
            return self.policy.default_max_hold_days
        return profile.expected_holding_days

    def half_life(self, dominant_alpha: str) -> float:
        profile = self.policy.alpha_profiles.get(dominant_alpha)
        if profile is None:
            return self.policy.default_half_life
        return profile.decay_half_life_days

    # ── Decay ─────────────────────────────────────────────────
    def apply_decay(
        self,
        position: PositionState,
        current_value: float,
        current_signal: Optional[EdgeCandidate] = None,
    ) -> float:
        """Apply time-decay multiplier to current_value.

        Args:
            position: PositionState (uses dominant_alpha + holding_days)
            current_value: pre-decay value (typically net_edge_5d)
            current_signal: optional fresh signal for reset condition

        Returns:
            decay-adjusted value
        """
        half_life = self.half_life(position.dominant_alpha)
        if half_life <= 0:
            return current_value

        decay = math.exp(-position.holding_days / half_life)

        # Reset floor when signal reconfirmed
        if current_signal is not None and self.is_reconfirmed(
            position, current_signal
        ):
            decay = max(decay, self.policy.reset_floor)

        return current_value * decay

    def is_reconfirmed(
        self,
        position: PositionState,
        signal: EdgeCandidate,
    ) -> bool:
        """Whether the current signal reconfirms the position thesis."""
        # Tier check
        if signal.edge_tier is None:
            return False
        signal_tier_rank = TIER_RANK_LOCAL.get(signal.edge_tier, -1)
        min_rank = TIER_RANK_LOCAL.get(self.policy.reset_min_tier, 2)
        if signal_tier_rank < min_rank:
            return False

        # Direction sign check
        if self.policy.require_direction_same_sign_for_reset:
            entry_sign = 1.0 if position.entry_edge >= 0 else -1.0
            signal_sign = 1.0 if signal.direction >= 0 else -1.0
            if entry_sign != signal_sign:
                return False

        return True

    # ── Static helper ─────────────────────────────────────────
    @staticmethod
    def dominant_alpha(directional_alphas: pd.Series) -> str:
        """Identify dominant alpha by absolute value (returns alpha name).

        Args:
            directional_alphas: pd.Series of {alpha_name: value}

        Returns:
            alpha name with maximum |value|. Empty string if input empty.
        """
        if directional_alphas is None or len(directional_alphas) == 0:
            return ""
        return str(directional_alphas.abs().idxmax())
