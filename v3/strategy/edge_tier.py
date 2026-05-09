"""V3.3 EdgeTierSystem — 분위수 기반 S/A/B/C tier (PR-2.3).

Pipeline position:
  EdgeEngine → net_edge_5d
       ↓
  EdgeTierSystem → S / A / B / C / BLOCKED (this module)
       ↓
  AllocationEngine (sizing 차등)

Tier semantics (V3.3_DESIGN.md §5.4):
  S — top 10% by net_edge AND high win_prob AND low |mae|: max sizing eligible
  A — net_edge ≥ 70th pct, win_prob ≥ 0.55: normal entry
  B — net_edge ≥ 50th pct: existing positions only (watch)
  C — below 50th pct: blocked
  BLOCKED — no net_edge (calibration failed)

Threshold derivation:
  derive_tier_thresholds() computes from calibration panel — NOT hardcoded.
  Stored in v3/config/edge_thresholds.json. Loaded via from_file().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.strategy._base import DecisionPolicy
from v3.strategy.types import EdgeCandidate, EdgeTier


# ──────────────────────────────────────────────────────────────
# TierThresholds
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TierThresholds:
    """Threshold values for S/A/B/C classification.

    All net_edge / win_prob thresholds are derived from calibration panel
    (분위수). Hardcoded defaults are starting points for tests.
    """
    # Net edge thresholds (분위수 도출)
    s_tier_net_edge: float = 0.0090       # ~90th pct
    a_tier_net_edge: float = 0.0040       # ~70th pct
    b_tier_net_edge: float = 0.0000       # ~50th pct

    # Win probability secondary criteria
    s_tier_winprob: float = 0.62
    a_tier_winprob: float = 0.55

    # Max acceptable expected MAE (less negative = better)
    s_tier_max_mae: float = -0.025        # 30th pct of MAE (less negative)


# ──────────────────────────────────────────────────────────────
# Threshold derivation from panel
# ──────────────────────────────────────────────────────────────
def derive_tier_thresholds(
    net_edges: Iterable[float],
    win_probs: Optional[Iterable[float]] = None,
    expected_maes: Optional[Iterable[float]] = None,
    s_pct: float = 0.90,
    a_pct: float = 0.70,
    b_pct: float = 0.50,
    s_winprob_pct: float = 0.70,
    s_mae_pct: float = 0.30,
    min_a_winprob: float = 0.55,
) -> TierThresholds:
    """Derive thresholds from calibration panel.

    Args:
        net_edges: panel net_edge_5d distribution
        win_probs: panel win_probability_5d distribution (optional, default
            falls to absolute thresholds)
        expected_maes: panel expected_mae_5d distribution (negative values).
            Pass raw values; we use 30th percentile = upper-30% of distribution
            (i.e., least negative — best 30%).
        s_pct/a_pct/b_pct: net_edge percentile cuts
        s_winprob_pct: S-tier win_prob percentile within top decile
        s_mae_pct: S-tier max_mae percentile (lower = stricter)
        min_a_winprob: floor on A-tier win_prob

    Returns:
        TierThresholds derived from panel.
    """
    arr = np.asarray(list(net_edges), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot derive tier thresholds from empty net_edges")

    s_ne = float(np.quantile(arr, s_pct))
    a_ne = float(np.quantile(arr, a_pct))
    b_ne = float(np.quantile(arr, b_pct))

    if win_probs is not None:
        wp_arr = np.asarray(list(win_probs), dtype=float)
        s_wp = float(np.quantile(wp_arr, s_winprob_pct)) if wp_arr.size > 0 else 0.62
    else:
        s_wp = 0.62

    if expected_maes is not None:
        mae_arr = np.asarray(list(expected_maes), dtype=float)
        # MAE is negative — 30th pct is the "best" (least negative) end
        s_mae = float(np.quantile(mae_arr, 1.0 - s_mae_pct)) if mae_arr.size > 0 else -0.025
    else:
        s_mae = -0.025

    return TierThresholds(
        s_tier_net_edge=s_ne,
        a_tier_net_edge=a_ne,
        b_tier_net_edge=b_ne,
        s_tier_winprob=s_wp,
        a_tier_winprob=min_a_winprob,
        s_tier_max_mae=s_mae,
    )


# ──────────────────────────────────────────────────────────────
# Threshold I/O
# ──────────────────────────────────────────────────────────────
def save_tier_thresholds(
    thresholds: TierThresholds,
    path: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "thresholds": {
            "s_tier_net_edge": thresholds.s_tier_net_edge,
            "a_tier_net_edge": thresholds.a_tier_net_edge,
            "b_tier_net_edge": thresholds.b_tier_net_edge,
            "s_tier_winprob": thresholds.s_tier_winprob,
            "a_tier_winprob": thresholds.a_tier_winprob,
            "s_tier_max_mae": thresholds.s_tier_max_mae,
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def load_tier_thresholds(
    path: str | Path,
) -> tuple[TierThresholds, dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    t = payload.get("thresholds", {})
    metadata = payload.get("metadata", {})
    return TierThresholds(
        s_tier_net_edge=float(t["s_tier_net_edge"]),
        a_tier_net_edge=float(t["a_tier_net_edge"]),
        b_tier_net_edge=float(t["b_tier_net_edge"]),
        s_tier_winprob=float(t["s_tier_winprob"]),
        a_tier_winprob=float(t["a_tier_winprob"]),
        s_tier_max_mae=float(t["s_tier_max_mae"]),
    ), metadata


# ──────────────────────────────────────────────────────────────
# EdgeTierSystem
# ──────────────────────────────────────────────────────────────
class EdgeTierSystem:
    """Assign S/A/B/C/BLOCKED tier based on net_edge + win_prob + |mae|.

    Implements DecisionPolicy protocol. Pure function.

    Tier criteria (all must hold for higher tier):
      S — net_edge ≥ s_threshold  AND
          win_probability ≥ s_winprob  AND
          expected_mae ≥ s_max_mae (less negative)
      A — net_edge ≥ a_threshold  AND
          win_probability ≥ a_winprob
      B — net_edge ≥ b_threshold
      C — net_edge < b_threshold
      BLOCKED — net_edge is None

    If S criteria for net_edge passes but win_prob/mae fail → demote to A.
    """

    name = "edge_tier"

    def __init__(
        self,
        thresholds: TierThresholds,
        enabled: bool = True,
    ):
        self.thresholds = thresholds
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        enabled: bool = True,
    ) -> "EdgeTierSystem":
        thresholds, _meta = load_tier_thresholds(path)
        return cls(thresholds=thresholds, enabled=enabled)

    def assign(self, candidate: EdgeCandidate) -> EdgeCandidate:
        """Assign tier to a single candidate. Pure function — returns new instance."""
        tier = self._compute_tier(candidate)
        return candidate.with_calibration(edge_tier=tier)

    def assign_batch(self, candidates: Iterable[EdgeCandidate]) -> list[EdgeCandidate]:
        return [self.assign(c) for c in candidates]

    def _compute_tier(self, c: EdgeCandidate) -> EdgeTier:
        # No calibration → BLOCKED
        if c.net_edge_5d is None:
            return "BLOCKED"

        net_edge = c.net_edge_5d
        win_prob = c.win_probability_5d if c.win_probability_5d is not None else 0.0
        mae = c.expected_mae_5d if c.expected_mae_5d is not None else 0.0

        # S-tier: must pass all three
        if (
            net_edge >= self.thresholds.s_tier_net_edge
            and win_prob >= self.thresholds.s_tier_winprob
            and mae >= self.thresholds.s_tier_max_mae
        ):
            return "S"

        # A-tier: net_edge AND win_prob (mae optional)
        if (
            net_edge >= self.thresholds.a_tier_net_edge
            and win_prob >= self.thresholds.a_tier_winprob
        ):
            return "A"

        # If net_edge would qualify A but win_prob doesn't → still A if just
        # net_edge passes A threshold (demote from S would land here)
        if net_edge >= self.thresholds.s_tier_net_edge:
            return "A"  # explicit demote from S

        # B-tier: net_edge above b_threshold
        if net_edge >= self.thresholds.b_tier_net_edge:
            return "B"

        return "C"


# ──────────────────────────────────────────────────────────────
# Tier ordering (for rotation/pyramid policies in later PRs)
# ──────────────────────────────────────────────────────────────
TIER_RANK: dict[str, int] = {
    "C": 0,
    "B": 1,
    "A": 2,
    "S": 3,
    "BLOCKED": -1,
}


def tier_rank(tier: Optional[str]) -> int:
    """Numeric rank for tier comparisons. BLOCKED < C."""
    if tier is None:
        return -1
    return TIER_RANK.get(tier, -1)
