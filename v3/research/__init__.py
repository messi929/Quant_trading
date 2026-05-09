"""V3.3 research scripts.

Modules:
  - build_edge_dataset (PR-1.4): N-year backtest replay → calibration panel
  - calibrate_edge (PR-2.1): panel → calibration_table.json
  - validate_edge (PR-2.1): OOS validation (decile monotonicity)
  - ablation_specs (PR-5.1): 16 ablation configurations
  - ablation_runner (PR-5.1): per-ablation execution + metrics
  - ablation_report (PR-5.1): comparison matrix + pass/fail markdown
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
from v3.research.calibrate_edge import (
    aggregate_bucket,
    assign_deciles_to_panel,
    build_calibration_table,
    calibrate_panel,
    compute_decile_breakpoints,
)
from v3.research.validate_edge import (
    DecileStats,
    ValidationResult,
    compute_decile_stats,
    render_markdown_report,
    validate_oos,
)
from v3.research.ablation_specs import (
    ABLATION_SPECS,
    AblationSpec,
    get_spec,
    list_ablations,
    load_features_from_yaml,
    save_specs_as_yaml,
    specs_summary,
)
from v3.research.ablation_runner import (
    AblationMetrics,
    AblationRunner,
    ActionDistribution,
    run_sweep,
)
from v3.research.ablation_report import (
    build_comparison_matrix,
    evaluate_ablation_pass,
    render_comparison_markdown,
    save_comparison_report,
)

__all__ = [
    # build_edge_dataset
    "BuildConfig",
    "ForwardOutcome",
    "PANEL_COLUMNS",
    "assign_liquidity_bucket",
    "assign_vol_state",
    "build_edge_dataset",
    "compute_forward_outcome",
    "panel_sanity_summary",
    "verify_no_lookahead",
    # calibrate_edge
    "aggregate_bucket",
    "assign_deciles_to_panel",
    "build_calibration_table",
    "calibrate_panel",
    "compute_decile_breakpoints",
    # validate_edge
    "DecileStats",
    "ValidationResult",
    "compute_decile_stats",
    "render_markdown_report",
    "validate_oos",
    # ablation_specs
    "ABLATION_SPECS",
    "AblationSpec",
    "get_spec",
    "list_ablations",
    "load_features_from_yaml",
    "save_specs_as_yaml",
    "specs_summary",
    # ablation_runner
    "AblationMetrics",
    "AblationRunner",
    "ActionDistribution",
    "run_sweep",
    # ablation_report
    "build_comparison_matrix",
    "evaluate_ablation_pass",
    "render_comparison_markdown",
    "save_comparison_report",
]
