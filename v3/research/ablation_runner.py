"""V3.3 Ablation Runner — execute ablations and collect metrics (PR-5.1).

Runs a single ablation by:
  1. Apply FeatureFlagsConfig override
  2. Construct BookOptimizer with all dependencies
  3. Iterate dates (synthetic or replay panel)
  4. Collect BookActions per date
  5. Aggregate metrics (return, win_rate, deployed, action distribution)

Synthetic vs panel mode:
  Synthetic: deterministic small dataset for unit testing (this file)
  Panel: production server replay with 5y data (separate runner script)

This module is the LIBRARY function. CLI integration is in
v3/scripts/run_ablation_sweep.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from v3.config.schema import FeatureFlagsConfig
from v3.research.ablation_specs import AblationSpec
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.types import BookAction


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ActionDistribution:
    """Frequency of each action_type across the run."""
    add_new: int = 0
    add_to_winner: int = 0
    keep: int = 0
    trim: int = 0
    exit: int = 0
    rotate: int = 0
    no_action: int = 0
    blocked: int = 0

    @property
    def total(self) -> int:
        return (
            self.add_new + self.add_to_winner + self.keep + self.trim
            + self.exit + self.rotate + self.no_action + self.blocked
        )


@dataclass(frozen=True)
class AblationMetrics:
    """Aggregate metrics for a single ablation run.

    Two source modes:
      Synthetic (BookOptimizer.decide stream): action_distribution + leverage
      Backtest (BacktestEngine.run): sharpe / max_drawdown / total_return etc.
    """
    ablation_name: str
    family: str
    n_dates: int
    n_actions: int
    action_distribution: ActionDistribution
    total_target_weight: float       # sum of ADD_NEW + ADD_TO_WINNER weights
    avg_deployed: float              # mean total weight per date (proxy for deploy %)
    max_single_weight: float
    leverage_violations: int         # Σ weights > 1.00 occurrences
    pyramid_count: int
    rotation_count: int
    risk_exit_count: int
    error_messages: tuple[str, ...] = field(default_factory=tuple)
    # PR-2.6: Backtest-specific (None for synthetic mode)
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    n_trades: Optional[int] = None


# ──────────────────────────────────────────────────────────────
# AblationRunner
# ──────────────────────────────────────────────────────────────
class AblationRunner:
    """Run a single ablation by invoking BookOptimizer per date.

    The runner is decision-level — it does not simulate fills, P&L, or
    realize positions. Output is a metrics summary of BookAction stream.

    For full backtest equity curve simulation, integrate via BacktestEngine
    (deferred to PR-5.2 or production server runs).

    Usage:
        runner = AblationRunner(
            book_optimizer_factory=lambda flags: BookOptimizer(...),
        )
        metrics = runner.run(
            spec=get_spec("A4_exit_thesis"),
            inputs_iter=...,  # iterable of (date, ohlcv, vol_scores, ...) per date
        )
    """

    def __init__(
        self,
        book_optimizer_factory: Callable[[FeatureFlagsConfig], BookOptimizer],
    ):
        """
        Args:
            book_optimizer_factory: callable taking FeatureFlagsConfig,
                returning a configured BookOptimizer with all dependencies
                wired (signal_gen, edge layers, exit policies, etc.)
        """
        self.factory = book_optimizer_factory

    def run(
        self,
        spec: AblationSpec,
        inputs_iter: Iterable[dict],
    ) -> AblationMetrics:
        """Execute ablation across iterable of date inputs.

        Args:
            spec: AblationSpec (defines features)
            inputs_iter: yields dict per date with keys
                {date, ohlcv, vol_scores, regime, cost, state, positions,
                 exit_triggers, ticker_volume_map, sector_map, ...}

        Returns:
            AblationMetrics summary.
        """
        bo = self.factory(spec.features)

        action_counts = {
            "ADD_NEW": 0, "ADD_TO_WINNER": 0, "KEEP": 0, "TRIM": 0,
            "EXIT": 0, "ROTATE": 0, "NO_ACTION": 0, "BLOCKED": 0,
        }
        all_actions: list[BookAction] = []
        date_total_weights: list[float] = []
        leverage_violations = 0
        risk_exit_count = 0
        errors: list[str] = []
        n_dates = 0

        for inp in inputs_iter:
            n_dates += 1
            try:
                actions = bo.decide(**self._normalize_inputs(inp))
            except Exception as exc:  # pragma: no cover
                errors.append(f"date={inp.get('date')}: {exc!r}")
                continue

            all_actions.extend(actions)
            for a in actions:
                action_counts[a.action_type] = (
                    action_counts.get(a.action_type, 0) + 1
                )
                if a.is_risk_exit:
                    risk_exit_count += 1

            # Per-date deployed weight check
            day_total = sum(
                a.target_weight for a in actions
                if a.action_type in ("ADD_NEW", "ADD_TO_WINNER")
            )
            date_total_weights.append(day_total)
            if day_total > 1.0 + 1e-6:
                leverage_violations += 1

        # Flush any buffered diagnostics
        bo.flush_diagnostics()

        max_single = max(
            (a.target_weight for a in all_actions
             if a.action_type in ("ADD_NEW", "ADD_TO_WINNER")),
            default=0.0,
        )
        avg_deployed = (
            float(np.mean(date_total_weights)) if date_total_weights else 0.0
        )

        return AblationMetrics(
            ablation_name=spec.name,
            family=spec.family,
            n_dates=n_dates,
            n_actions=len(all_actions),
            action_distribution=ActionDistribution(
                add_new=action_counts["ADD_NEW"],
                add_to_winner=action_counts["ADD_TO_WINNER"],
                keep=action_counts["KEEP"],
                trim=action_counts["TRIM"],
                exit=action_counts["EXIT"],
                rotate=action_counts["ROTATE"],
                no_action=action_counts["NO_ACTION"],
                blocked=action_counts["BLOCKED"],
            ),
            total_target_weight=sum(date_total_weights),
            avg_deployed=avg_deployed,
            max_single_weight=max_single,
            leverage_violations=leverage_violations,
            pyramid_count=action_counts["ADD_TO_WINNER"],
            rotation_count=action_counts["ROTATE"],
            risk_exit_count=risk_exit_count,
            error_messages=tuple(errors),
        )

    @staticmethod
    def _normalize_inputs(inp: dict) -> dict:
        """Pass-through with sensible defaults for missing keys."""
        out = dict(inp)
        out.setdefault("ticker_volume_map", {})
        out.setdefault("sector_map", {})
        out.setdefault("ticker_vol_map", {})
        out.setdefault("liquidity_bucket_map", {})
        out.setdefault("positions", [])
        out.setdefault("exit_triggers", {})
        out.setdefault("signal_age_hours", 0.0)
        out.setdefault("rotations_this_month", 0)
        out.setdefault("adds_per_position", {})
        out.setdefault("initial_weights", {})
        # Drop the date key — BookOptimizer uses as_of internally
        if "date" in out and "as_of" not in out:
            out["as_of"] = out.pop("date")
        elif "date" in out:
            out.pop("date")
        return out


# ──────────────────────────────────────────────────────────────
# Multi-ablation sweep
# ──────────────────────────────────────────────────────────────
def run_sweep(
    specs: list[AblationSpec],
    runner: AblationRunner,
    inputs_factory: Callable[[], Iterable[dict]],
) -> list[AblationMetrics]:
    """Run multiple ablations sequentially.

    Args:
        specs: list of AblationSpec
        runner: AblationRunner instance
        inputs_factory: callable returning fresh iterable of inputs per ablation
            (each spec needs same input sequence)

    Returns:
        list of AblationMetrics in same order as specs.
    """
    results: list[AblationMetrics] = []
    for spec in specs:
        logger.info(f"Running ablation: {spec.name}")
        inputs = inputs_factory()
        metrics = runner.run(spec, inputs)
        results.append(metrics)
        logger.info(
            f"  {spec.name}: {metrics.n_actions} actions, "
            f"avg_deployed={metrics.avg_deployed:.3f}, "
            f"pyramid={metrics.pyramid_count}, rotation={metrics.rotation_count}, "
            f"violations={metrics.leverage_violations}"
        )
    return results
