"""V3.3 EdgeEngine — calibrated expected_return → net_edge_5d.

Pipeline position:
  raw_opportunity → EdgeCalibrator → expected_return_5d
                                  ↓
                                EdgeEngine (this module)
                                  ↓
                                net_edge_5d
                                  ↓
                                EdgeTierSystem → S/A/B/C tier

Formula:
  net_edge_5d = expected_return_5d
              - expected_cost
              - slippage_buffer
              - downside_risk_penalty

  expected_cost   = commission_roundtrip + spread_estimate + market_impact_base
  slippage_buffer = clip(min_slip + ticker_vol × multiplier, [min_slip, max_slip])
  downside_risk_penalty = lambda_mae × |expected_mae_5d|

  entry_pass = net_edge_5d > entry_threshold

Calibration trust shrinkage (V3.3_DESIGN §4.2):
  if calibration_n < calibration_n_demote (default 100):
    net_edge *= 0.5

entry_threshold derivation:
  Use derive_entry_threshold(panel, percentile=0.70) to compute from
  calibration panel — NOT hardcoded. Stored in edge_thresholds.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.strategy.types import EdgeCandidate


# ──────────────────────────────────────────────────────────────
# Cost model — V3.2.1 0.001 단일값 → 분해
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CostModel:
    """Decomposed cost model. V3.2.1 통합 0.001 → 4 components.

    NASDAQ defaults sum to ~0.0010 + vol-dependent slippage buffer.
    """
    commission_roundtrip: float = 0.0002
    spread_estimate: float = 0.0003
    market_impact_base: float = 0.0005
    slippage_vol_multiplier: float = 0.10
    min_slippage_buffer: float = 0.0010
    max_slippage_buffer: float = 0.0040

    def expected_cost(self, ticker: str = "") -> float:
        """Per-trade expected cost (commission + spread + impact)."""
        return (
            self.commission_roundtrip
            + self.spread_estimate
            + self.market_impact_base
        )

    def slippage_buffer(self, ticker: str, ticker_vol: float) -> float:
        """Vol-scaled slippage buffer, clipped to [min, max]."""
        buf = self.min_slippage_buffer + max(ticker_vol, 0.0) * self.slippage_vol_multiplier
        return float(np.clip(buf, self.min_slippage_buffer, self.max_slippage_buffer))


# ──────────────────────────────────────────────────────────────
# Edge policy — entry threshold + risk penalty
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EdgePolicy:
    """Edge engine policy parameters.

    entry_threshold: load from derive_entry_threshold() (분위수 도출).
    The 0.0040 default is a starting point only — ROADMAP §5 Phase 2 ablation
    requires deriving from calibration panel, not hardcoding.
    """
    horizon_days: int = 5
    entry_threshold: float = 0.0040
    high_conviction_threshold: float = 0.0090
    lambda_mae: float = 0.20
    calibration_n_demote: int = 100
    demote_factor: float = 0.5


# ──────────────────────────────────────────────────────────────
# Entry threshold derivation (분위수 → starting point)
# ──────────────────────────────────────────────────────────────
def derive_entry_threshold(
    net_edges: Iterable[float],
    percentile: float = 0.70,
) -> float:
    """Derive entry_threshold from calibrated panel net_edge distribution.

    Args:
        net_edges: panel net_edge_5d values
        percentile: top fraction kept. 0.70 means threshold = 70th percentile
            (top 30% pass).

    Returns:
        Threshold value at given percentile.
    """
    arr = np.asarray(list(net_edges), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot derive threshold from empty net_edges")
    if not (0.0 < percentile < 1.0):
        raise ValueError(f"percentile must be in (0, 1): {percentile}")
    return float(np.quantile(arr, percentile))


# ──────────────────────────────────────────────────────────────
# EdgeEngine
# ──────────────────────────────────────────────────────────────
class EdgeEngine:
    """expected_return_5d → net_edge_5d (cost + risk 차감).

    Pure function. compute() returns new EdgeCandidate (frozen).

    Usage:
        eng = EdgeEngine(cost_model=CostModel(), policy=EdgePolicy())
        result = eng.compute(candidate, ticker_vol=0.25)
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        policy: Optional[EdgePolicy] = None,
    ):
        self.cost_model = cost_model or CostModel()
        self.policy = policy or EdgePolicy()

    def compute(
        self,
        candidate: EdgeCandidate,
        ticker_vol: float = 0.20,
    ) -> EdgeCandidate:
        """Compute net_edge_5d from calibrated candidate.

        Args:
            candidate: must have expected_return_5d + expected_mae_5d populated
                (i.e., already passed through EdgeCalibrator)
            ticker_vol: annualized vol for slippage buffer calc

        Returns:
            New EdgeCandidate with cost / risk / net_edge fields populated.
        """
        # Pre-condition: calibration already done
        if candidate.expected_return_5d is None or candidate.expected_mae_5d is None:
            return candidate.with_calibration(
                entry_pass=False,
                reject_reason="not_calibrated",
            )

        cost = self.cost_model.expected_cost(candidate.ticker)
        slip = self.cost_model.slippage_buffer(candidate.ticker, ticker_vol)
        downside = self.policy.lambda_mae * abs(candidate.expected_mae_5d)

        net_edge = (
            candidate.expected_return_5d - cost - slip - downside
        )

        # Calibration confidence shrinkage
        if (
            candidate.calibration_n is not None
            and candidate.calibration_n < self.policy.calibration_n_demote
        ):
            net_edge *= self.policy.demote_factor

        passes = net_edge > self.policy.entry_threshold

        return candidate.with_calibration(
            expected_cost=cost,
            slippage_buffer=slip,
            downside_risk_penalty=downside,
            net_edge_5d=net_edge,
            entry_pass=passes,
            reject_reason=None if passes else "net_edge_below_threshold",
        )

    def compute_batch(
        self,
        candidates: Iterable[EdgeCandidate],
        ticker_vols: Optional[dict[str, float]] = None,
        default_vol: float = 0.20,
    ) -> list[EdgeCandidate]:
        """Compute net_edge for multiple candidates."""
        ticker_vols = ticker_vols or {}
        return [
            self.compute(c, ticker_vol=ticker_vols.get(c.ticker, default_vol))
            for c in candidates
        ]
