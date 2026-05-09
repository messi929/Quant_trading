"""Run V3.3 ablation sweep + comparison + promotion plan (PR-5.2).

Production usage requires:
  - Trained VolTransformer + alpha_weights.json
  - 5y OHLCV + macro panel (build_edge_dataset.py)
  - Calibration table (calibrate_edge.py)

This CLI is a stub that demonstrates the wiring. Actual sweep runs from
server with full data dependencies. Synthetic mode (--synthetic) runs
against mock data for sanity checking.

Outputs:
  research/reports/ablation_<date>_comparison.md
  research/reports/ablation_<date>_promotion.md

CLI:
  $PYTHON v3/scripts/run_ablation_sweep.py \\
      --output-dir research/reports \\
      --tag 2026-09 \\
      [--synthetic]

  # In production (server):
  $PYTHON v3/scripts/run_ablation_sweep.py \\
      --output-dir /opt/quant/research/reports \\
      --tag 2026-09 \\
      --calibration v3/config/edge_calibration_2026-09.json \\
      --panel data/research/edge_dataset_2026-09.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import pandas as pd
from loguru import logger

from v3.config.schema import FeatureFlagsConfig, V3Config, load_config
from v3.research.ablation_runner import (
    AblationMetrics,
    AblationRunner,
    ActionDistribution,
    run_sweep,
)
from v3.research.ablation_report import save_comparison_report
from v3.research.ablation_specs import (
    ABLATION_SPECS,
    AblationSpec,
    apply_features_to_config,
)
from v3.research.promotion_decision import (
    build_promotion_plan,
    save_promotion_plan,
)
from v3.rules.entry import OperationalState
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.regime_v2 import Regime
from v3.strategy.signal import TradeSignal


# ──────────────────────────────────────────────────────────────
# Synthetic mode helpers (sanity check)
# ──────────────────────────────────────────────────────────────
def _synthetic_signal() -> TradeSignal:
    return TradeSignal(
        action="TRADE",
        positions=[
            {"ticker": "AAPL", "weight": 0.30, "direction": 0.05,
             "conviction": 0.7, "opportunity": 0.0035},
            {"ticker": "MSFT", "weight": 0.20, "direction": 0.03,
             "conviction": 0.6, "opportunity": 0.0018},
        ],
        cash_weight=0.50,
        regime_name="neutral",
        regime_score=0.5,
        position_scale=0.9,
        n_candidates=5,
        n_approved=2,
        rejection_reasons={"net_edge": 3},
        opportunity_map={
            "AAPL": 0.0035, "MSFT": 0.0018,
            "GOOG": 0.0010, "AMZN": 0.0005, "META": 0.0003,
        },
        opportunity_gate=0.00175,
    )


def _synthetic_factory():
    """Synthetic BookOptimizer factory for sanity sweep.

    All Phase 3+4 dependencies are None. Only Edge feature flags work in
    pass-through manner.
    """
    sg = MagicMock()
    sg.generate.return_value = _synthetic_signal()

    def factory(features: FeatureFlagsConfig) -> BookOptimizer:
        return BookOptimizer(signal_gen=sg, features=features)

    return factory


def _synthetic_inputs(n_dates: int = 20) -> Iterable[dict]:
    regime = Regime(
        name="neutral", score=0.5,
        alpha_weights={"trend": 1.0, "reversion": 0.0},
        position_scale=0.9, confidence=1.0,
        detected_at=pd.Timestamp("2026-05-09"),
    )
    state = OperationalState(
        current_positions=0, monthly_trades=0,
        recent_win_rate=0.5, current_mdd=0.0,
        circuit_breaker_active=False,
    )
    for i in range(n_dates):
        yield {
            "date": pd.Timestamp("2026-05-09") + pd.Timedelta(days=i),
            "ohlcv": pd.DataFrame(),
            "vol_scores": pd.DataFrame(),
            "regime": regime,
            "cost": 0.001,
            "state": state,
        }


# ──────────────────────────────────────────────────────────────
# Production mode — BacktestEngine wiring
# ──────────────────────────────────────────────────────────────
def _backtest_to_ablation_metrics(
    bt_result, spec: AblationSpec
) -> AblationMetrics:
    """Convert BacktestResult → AblationMetrics.

    BacktestEngine populates Sharpe / MDD / Return / WinRate / PF.
    Action distribution는 trade count 기반 근사 (BookAction 직접 trace는
    BookOptimizer.decide_with_context 호출 시점에서 가능 — 별도 PR).
    """
    return AblationMetrics(
        ablation_name=spec.name,
        family=spec.family,
        n_dates=len(bt_result.equity_curve) if hasattr(bt_result, "equity_curve") else 0,
        n_actions=int(bt_result.total_trades),
        action_distribution=ActionDistribution(
            add_new=int(bt_result.total_trades),
        ),
        total_target_weight=0.0,    # not tracked by current BacktestEngine
        avg_deployed=0.0,           # not tracked
        max_single_weight=0.0,
        leverage_violations=0,      # backtest engine clamps to ≤1.00
        pyramid_count=0,
        rotation_count=0,
        risk_exit_count=sum(
            1 for t in bt_result.trades
            if any(k in t.exit_reason.lower() for k in
                   ("stop", "circuit", "vol_contraction"))
        ),
        # Backtest-specific metrics
        sharpe=float(bt_result.sharpe),
        max_drawdown=float(bt_result.max_drawdown),
        total_return=float(bt_result.total_return),
        win_rate=float(bt_result.win_rate),
        profit_factor=float(bt_result.profit_factor),
        n_trades=int(bt_result.total_trades),
    )


def _run_production_sweep(args) -> list[AblationMetrics]:
    """Run ablation sweep using BacktestEngine for each spec.

    Each ablation gets its own BacktestEngine instance with apply_features_to_config
    applied. BookOptimizer routing inside BacktestEngine activates the spec features.

    Limitations:
      - Phase 3+4 effects (exit_thesis, pyramid, rotation) require
        BacktestEngine to pass `positions=...` into BookOptimizer (PR-2.7).
      - Until then, Phase 2 Edge layer effects (calibrator/engine/tier) are
        the primary measurable differential.
    """
    from v3.backtest.engine import BacktestEngine

    base_cfg = load_config()

    logger.info(f"Loading panel from {args.panel}")
    df = pd.read_parquet(args.panel)

    # vol_predictions: assumed to be a separate parquet or derived; here we
    # require a paired file or skip the ablation if missing
    vol_pred_path = args.panel.parent / f"{args.panel.stem}_vol_predictions.parquet"
    if not vol_pred_path.exists():
        logger.error(
            f"vol_predictions parquet not found at {vol_pred_path}. "
            "Generate via VolInference batch and place alongside panel."
        )
        return []
    vol_predictions = pd.read_parquet(vol_pred_path)

    results: list[AblationMetrics] = []
    for spec in ABLATION_SPECS:
        logger.info(f"Running ablation: {spec.name}")
        try:
            cfg_for_spec = apply_features_to_config(base_cfg, spec)
            engine = BacktestEngine(cfg_for_spec)
            bt_result = engine.run(df, vol_predictions)
            metrics = _backtest_to_ablation_metrics(bt_result, spec)
            results.append(metrics)
            logger.info(
                f"  {spec.name}: Sharpe={metrics.sharpe:.2f}, "
                f"Return={metrics.total_return:.3f}, "
                f"MDD={metrics.max_drawdown:.3f}, "
                f"trades={metrics.n_trades}"
            )
        except Exception as exc:
            logger.error(f"  {spec.name} FAILED: {exc!r}")
            # Append placeholder so report still shows the failed ablation
            results.append(AblationMetrics(
                ablation_name=spec.name,
                family=spec.family,
                n_dates=0, n_actions=0,
                action_distribution=ActionDistribution(),
                total_target_weight=0.0, avg_deployed=0.0,
                max_single_weight=0.0,
                leverage_violations=0,
                pyramid_count=0, rotation_count=0, risk_exit_count=0,
                error_messages=(str(exc),),
            ))

    return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("research/reports"),
        help="Directory for report markdown files",
    )
    p.add_argument(
        "--tag", type=str,
        default=pd.Timestamp.now().strftime("%Y%m%d"),
        help="Tag suffix for report filenames",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic factory + inputs (sanity mode)",
    )
    p.add_argument(
        "--n-dates", type=int, default=20,
        help="Number of synthetic dates (synthetic mode only)",
    )
    p.add_argument(
        "--calibration", type=Path, default=None,
        help="Path to calibration_table.json (production mode)",
    )
    p.add_argument(
        "--panel", type=Path, default=None,
        help="Path to edge_dataset.parquet (production mode)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        logger.info(
            f"Running ablation sweep in SYNTHETIC mode "
            f"({args.n_dates} dates, {len(ABLATION_SPECS)} ablations)"
        )
        factory = _synthetic_factory()
        inputs_factory = lambda: _synthetic_inputs(n_dates=args.n_dates)
        runner = AblationRunner(book_optimizer_factory=factory)
        results = run_sweep(
            ABLATION_SPECS,
            runner=runner,
            inputs_factory=inputs_factory,
        )
    else:
        # Production mode — BacktestEngine for each ablation
        if args.panel is None:
            logger.error(
                "--panel <path/to/edge_dataset.parquet> required in "
                "production mode. Use --synthetic for sanity check."
            )
            return 1
        results = _run_production_sweep(args)

    if not results:
        logger.error("No results — sweep aborted")
        return 1

    # Comparison report
    comparison_path = args.output_dir / f"ablation_{args.tag}_comparison.md"
    save_comparison_report(results, comparison_path)

    # Promotion plan
    plan = build_promotion_plan(results)
    promotion_path = args.output_dir / f"ablation_{args.tag}_promotion.md"
    save_promotion_plan(plan, promotion_path)

    logger.info(
        f"Ablation sweep complete:\n"
        f"  - {comparison_path}\n"
        f"  - {promotion_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
