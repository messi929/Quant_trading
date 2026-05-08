"""V3.3 research scripts.

Modules:
  - build_edge_dataset (PR-1.4): N-year backtest replay → calibration panel
  - calibrate_edge (PR-2.1): panel → calibration_table.json + tier_thresholds
  - validate_edge (PR-2.1): OOS validation (decile/tier monotonicity)
"""

from v3.research.build_edge_dataset import (
    BuildConfig,
    ForwardOutcome,
    PANEL_COLUMNS,
    assign_liquidity_bucket,
    assign_vol_state,
    build_edge_dataset,
    compute_forward_outcome,
    panel_sanity_summary,
    verify_no_lookahead,
)

__all__ = [
    "BuildConfig",
    "ForwardOutcome",
    "PANEL_COLUMNS",
    "assign_liquidity_bucket",
    "assign_vol_state",
    "build_edge_dataset",
    "compute_forward_outcome",
    "panel_sanity_summary",
    "verify_no_lookahead",
]
