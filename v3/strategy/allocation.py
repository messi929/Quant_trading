"""V3.3 AllocationEngine — net_edge / risk 기반 sizing (PR-4.1).

Pipeline position:
  EdgeTierSystem → tier-assigned candidates
       ↓
  AllocationEngine (this module)
       ↓
  {ticker: target_weight}, leverage ≤ 1.00 strict

핵심 변화 (V3.2.1 → V3.3):
  V3.2.1: confidence 기반 sizing (VolTargetSizer + Half-Kelly)
  V3.3:   net_edge / |expected_mae| 기반 raw_weight
          × regime_exposure_scale × correlation_drag × liquidity
          → clamp [min_position_weight, max_single_weight]
          → normalize to budget.max_gross_exposure (max 1.00)

VolTargetSizer 와의 관계:
  AllocationEngine은 VolTargetSizer를 대체하지 않고 보완.
  AllocationEngine.allocate()의 1차 입력이 net_edge/risk이고, sizer의
  Half-Kelly + correlation drag를 overlay로 적용.

Leverage policy (영구):
  max_gross_exposure 1.00 강제 cap. budget이 1.20 등 leverage value를
  가지면 1.00으로 clip. conviction-or-cash 원칙 (페르소나 §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from v3.strategy.regime_v2 import Regime
from v3.strategy.types import EdgeCandidate


# Leverage 거부 — 절대 cap
LEVERAGE_CAP: float = 1.00


# ──────────────────────────────────────────────────────────────
# RegimeBudget — per-regime exposure + vol target
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RegimeBudget:
    """Per-regime exposure budget. max_gross_exposure ≤ 1.00 always."""
    max_gross_exposure: float
    target_annual_vol: float


def _default_regime_budgets() -> dict[str, RegimeBudget]:
    """Default regime budgets — leverage rejected."""
    return {
        "strong_bull": RegimeBudget(max_gross_exposure=1.00, target_annual_vol=0.20),
        "bull":        RegimeBudget(max_gross_exposure=0.90, target_annual_vol=0.18),
        "neutral":     RegimeBudget(max_gross_exposure=0.70, target_annual_vol=0.15),
        "caution":     RegimeBudget(max_gross_exposure=0.40, target_annual_vol=0.10),
        "bear":        RegimeBudget(max_gross_exposure=0.00, target_annual_vol=0.00),
    }


# ──────────────────────────────────────────────────────────────
# PositionConstraints
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PositionConstraints:
    """Per-position structural constraints."""
    max_positions: int = 3
    max_single_weight: float = 0.40
    min_position_weight: float = 0.15        # Phase 25.2 sizer floor
    max_sector_weight: float = 0.60
    volume_participation_max: float = 0.05
    min_order_amount_krw: float = 5_000_000


# ──────────────────────────────────────────────────────────────
# AllocationEngine
# ──────────────────────────────────────────────────────────────
class AllocationEngine:
    """net_edge / |expected_mae| 기반 portfolio sizing.

    Pure function. allocate() returns {ticker: weight} dict.

    Leverage 거부 (영구):
      모든 weight 합 ≤ min(regime_budget, LEVERAGE_CAP) = min(.., 1.00).

    Pipeline:
      1. Filter: entry_pass=True AND net_edge > 0
      2. Sort by net_edge desc, take top max_positions
      3. Compute edge_risk_ratio = net_edge / max(|expected_mae|, 0.005)
      4. raw_weight = ratio / sum(ratios)
      5. Apply regime_exposure_scale (capped at LEVERAGE_CAP)
      6. (Optional) Correlation drag from correlation_matrix
      7. Clamp [min_position_weight, max_single_weight]
      8. Normalize to budget.max_gross_exposure if overshooting
      9. Apply sector cap (post-hoc reduction)
    """

    LEVERAGE_CAP = LEVERAGE_CAP  # class attribute for tests

    def __init__(
        self,
        constraints: Optional[PositionConstraints] = None,
        regime_budgets: Optional[dict[str, RegimeBudget]] = None,
        min_downside_floor: float = 0.005,
    ):
        self.constraints = constraints or PositionConstraints()
        self.regime_budgets = regime_budgets or _default_regime_budgets()
        self.min_downside_floor = min_downside_floor

    # ── Public API ────────────────────────────────────────────
    def allocate(
        self,
        candidates: list[EdgeCandidate],
        regime: Regime,
        sector_map: Optional[dict[str, str]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute target weights.

        Args:
            candidates: EdgeCandidate list (post-EdgeEngine, with net_edge_5d
                + expected_mae_5d populated)
            regime: current Regime (provides budget)
            sector_map: {ticker: sector} for sector cap
            correlation_matrix: per-candidate correlation (n × n) for drag

        Returns:
            {ticker: target_weight} or empty dict if no candidates pass.
            Σ weights ≤ min(budget.max_gross_exposure, LEVERAGE_CAP).
        """
        sector_map = sector_map or {}

        budget = self.regime_budgets.get(regime.name)
        if budget is None or budget.max_gross_exposure <= 0:
            # Bear / unknown → CASH
            return {}

        # Step 1: filter passing candidates with positive net_edge
        passing = [
            c for c in candidates
            if c.entry_pass and c.net_edge_5d is not None and c.net_edge_5d > 0
        ]
        if not passing:
            return {}

        # Step 2: sort by net_edge, take top max_positions
        passing.sort(key=lambda c: c.net_edge_5d or 0.0, reverse=True)
        passing = passing[: self.constraints.max_positions]

        # Step 3: edge / risk ratios
        ratios: dict[str, float] = {}
        for c in passing:
            downside = max(abs(c.expected_mae_5d or 0.0), self.min_downside_floor)
            ratios[c.ticker] = (c.net_edge_5d or 0.0) / downside

        total_ratio = sum(ratios.values())
        if total_ratio <= 0:
            return {}

        raw_weights = {t: r / total_ratio for t, r in ratios.items()}

        # Step 5: regime exposure budget (with leverage cap)
        exposure = min(budget.max_gross_exposure, self.LEVERAGE_CAP)
        scaled = {t: w * exposure for t, w in raw_weights.items()}

        # Step 6: (optional) correlation drag
        if correlation_matrix is not None:
            scaled = self._apply_correlation_drag(scaled, passing, correlation_matrix)

        # Step 7: clamp [min, max] per position
        clamped: dict[str, float] = {}
        for ticker, w in scaled.items():
            if w < self.constraints.min_position_weight:
                # Below floor — drop (don't take meaningless tiny positions)
                continue
            clamped[ticker] = min(w, self.constraints.max_single_weight)

        # Step 8: normalize to budget if overshooting
        total = sum(clamped.values())
        if total > exposure and total > 0:
            scale = exposure / total
            clamped = {t: w * scale for t, w in clamped.items()}

        # Step 9: sector cap
        clamped = self._apply_sector_cap(clamped, passing, sector_map)

        # Final leverage assertion
        final_total = sum(clamped.values())
        assert final_total <= self.LEVERAGE_CAP + 1e-9, (
            f"AllocationEngine produced leverage {final_total:.4f} > "
            f"LEVERAGE_CAP {self.LEVERAGE_CAP} — invariant violation"
        )

        return clamped

    # ── Helpers ───────────────────────────────────────────────
    def _apply_correlation_drag(
        self,
        weights: dict[str, float],
        candidates: list[EdgeCandidate],
        corr: np.ndarray,
    ) -> dict[str, float]:
        """Reduce co-moving positions (Ledoit-Wolf style shrinkage applied
        to off-diagonal). High avg correlation → drag factor < 1."""
        n = len(candidates)
        if n != corr.shape[0] or n != corr.shape[1]:
            logger.warning(
                f"AllocationEngine: correlation matrix shape mismatch "
                f"({corr.shape} vs {n} candidates) — skipping drag"
            )
            return weights

        result: dict[str, float] = {}
        for i, c in enumerate(candidates):
            avg_corr_off_diag = (corr[i, :].sum() - corr[i, i]) / max(n - 1, 1)
            drag = max(0.5, 1.0 - avg_corr_off_diag * 0.3)
            result[c.ticker] = weights.get(c.ticker, 0.0) * drag
        return result

    def _apply_sector_cap(
        self,
        weights: dict[str, float],
        candidates: list[EdgeCandidate],
        sector_map: dict[str, str],
    ) -> dict[str, float]:
        """Reduce ticker weights if sector total exceeds max_sector_weight."""
        if not weights or not sector_map:
            return weights

        # Group by sector
        ticker_sectors = {c.ticker: sector_map.get(c.ticker, "unknown") for c in candidates}
        sector_totals: dict[str, float] = {}
        for t, w in weights.items():
            s = ticker_sectors.get(t, "unknown")
            sector_totals[s] = sector_totals.get(s, 0.0) + w

        # Find overexposed sectors
        result = dict(weights)
        for sector, total in sector_totals.items():
            if total <= self.constraints.max_sector_weight:
                continue
            scale = self.constraints.max_sector_weight / total
            for t, s in ticker_sectors.items():
                if s == sector and t in result:
                    result[t] *= scale

        return result
