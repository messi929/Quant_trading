"""V3.3 immutable data types.

All domain objects in V3.3 are frozen dataclasses (Phase 2 원칙: no mutation).
Functional update via `replace()` or domain helpers like `with_calibration()`.

Object lifecycle:
  OpportunityScorer → EdgeCandidate (raw signal populated)
       ↓ EdgeCalibrator.calibrate()
  EdgeCandidate (expected_return populated)
       ↓ EdgeEngine.compute()
  EdgeCandidate (net_edge populated)
       ↓ EdgeTierSystem.assign()
  EdgeCandidate (tier populated)
       ↓ AllocationEngine / ExitThesis / Pyramid / Rotation / PartialExit
  BookAction (final decision)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import pandas as pd


EdgeTier = Literal["S", "A", "B", "C", "BLOCKED"]

ActionType = Literal[
    "ADD_NEW", "ADD_TO_WINNER", "KEEP", "TRIM",
    "EXIT", "ROTATE", "NO_ACTION", "BLOCKED",
]


# ──────────────────────────────────────────────────────────────
# EdgeCandidate
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EdgeCandidate:
    """Per-ticker signal enriched through Edge layer.

    Optional fields are populated incrementally:
      1. OpportunityScorer fills: direction, conviction, raw_opportunity
      2. EdgeCalibrator fills: expected_return_5d, expected_mae_5d, ...
      3. EdgeEngine fills: net_edge_5d, expected_cost, ...
      4. EdgeTierSystem fills: edge_tier
    """
    # Identity
    date: pd.Timestamp
    ticker: str

    # Raw signal (from OpportunityScorer)
    direction: float
    conviction: float
    raw_opportunity: float

    # Context
    regime: str
    regime_score: float
    sector: Optional[str] = None
    liquidity_bucket: Optional[str] = None
    vol_state: Optional[str] = None
    dominant_alpha: Optional[str] = None

    # Calibrated edge (from EdgeCalibrator)
    expected_return_5d: Optional[float] = None
    expected_mae_5d: Optional[float] = None
    expected_mfe_5d: Optional[float] = None
    win_probability_5d: Optional[float] = None
    calibration_n: Optional[int] = None
    calibration_key: Optional[tuple] = None

    # Net edge (from EdgeEngine)
    expected_cost: Optional[float] = None
    slippage_buffer: Optional[float] = None
    downside_risk_penalty: Optional[float] = None
    net_edge_5d: Optional[float] = None

    # Tier (from EdgeTierSystem)
    edge_tier: Optional[EdgeTier] = None

    # Decision
    entry_pass: bool = False
    reject_reason: Optional[str] = None

    def with_calibration(self, **kwargs) -> "EdgeCandidate":
        """Functional update — returns new instance with given fields replaced."""
        return replace(self, **kwargs)


# ──────────────────────────────────────────────────────────────
# PositionState
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PositionState:
    """Immutable snapshot of a held position.

    Replaces V3.2.1's mutable position dict for V3.3 decision flow.
    Bridge from legacy via PositionState.from_legacy_position() (later PRs).
    """
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    current_price: float
    current_weight: float

    # Edge tracking
    entry_edge: float
    residual_edge: float
    entry_tier: str
    current_tier: str

    # Performance
    pnl_pct: float
    max_unrealized_pnl_pct: float
    drawdown_from_peak_pct: float

    # Time
    holding_days: int
    dominant_alpha: str
    expected_holding_days: int

    # Thesis
    thesis_alive: bool
    thesis_break_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# BookAction
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BookAction:
    """Single decision unit emitted by BookOptimizer.

    All position-level and entry-level decisions normalize to BookAction.
    Live executor + backtest engine consume the same action stream.
    """
    date: pd.Timestamp
    action_type: ActionType
    ticker: str
    target_weight: float
    current_weight: float
    reason: str
    source_policy: str
    expected_impact: Optional[float] = None
    replacement_ticker: Optional[str] = None

    @property
    def weight_delta(self) -> float:
        return self.target_weight - self.current_weight

    @property
    def is_risk_exit(self) -> bool:
        return self.action_type == "EXIT" and "risk" in self.reason.lower()


# ──────────────────────────────────────────────────────────────
# NoTradeLog
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NoTradeLog:
    """Single rejection event for diagnostics."""
    date: pd.Timestamp
    ticker: str
    stage: str               # "opportunity" | "calibration" | "edge" | "tier" | "operational"
    reject_reason: str
    raw_opportunity: float
    regime: str
    net_edge_5d: Optional[float] = None
    edge_tier: Optional[str] = None
    details: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# TCSnapshot (Transfer Coefficient diagnostic — PR-1.2)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TCSnapshot:
    """Daily transfer coefficient + top decile capture snapshot.

    transfer_coefficient: Spearman correlation between net_edge ranking
        and final_weight ranking. ∈ [-1, 1]. Higher = better signal-to-weight.
    top_decile_capture: weight in top edge decile / total deployed weight.
        ∈ [0, 1]. 1.0 = perfect concentration on best signals.
    """
    date: pd.Timestamp
    transfer_coefficient: float
    top_decile_capture: float
    n_candidates: int
    n_deployed: int
    total_deployed_weight: float


# ──────────────────────────────────────────────────────────────
# CalibrationBucket
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CalibrationBucket:
    """Single calibration table cell.

    Built by v3/research/calibrate_edge.py from edge_dataset.parquet.
    Loaded by EdgeCalibrator at runtime.
    """
    key: tuple                      # ("global",) | (decile,) | (decile, regime) | ...
    n: int
    mean_forward_return_5d: float
    mean_mae_5d: float
    mean_mfe_5d: float
    win_rate_5d: float
    std_forward_return_5d: float
    median_forward_return_5d: float

    @property
    def standard_error(self) -> float:
        if self.n <= 0:
            return float("inf")
        return self.std_forward_return_5d / (self.n ** 0.5)

    def is_reliable(self, min_n: int = 30) -> bool:
        return self.n >= min_n
