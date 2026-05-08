"""V3.3 calibrate_edge.py tests (PR-2.1).

Verifies:
  - compute_decile_breakpoints (9 quantiles)
  - assign_deciles_to_panel (vectorized)
  - aggregate_bucket (mean/median/win_rate/PF)
  - build_calibration_table (4 fallback levels)
  - calibrate_panel end-to-end (synthetic panel → JSON)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.research.calibrate_edge import (
    aggregate_bucket,
    assign_deciles_to_panel,
    build_calibration_table,
    calibrate_panel,
    compute_decile_breakpoints,
)
from v3.strategy.edge_calibrator import load_calibration_table


def _make_panel(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "ticker": [f"T{i:03d}" for i in range(n)],
        "raw_opportunity": rng.normal(0.002, 0.005, n),
        "regime": rng.choice(
            ["neutral", "bull", "caution"], size=n, p=[0.6, 0.3, 0.1]
        ),
        "liquidity_bucket": rng.choice(["high", "mid", "low"], size=n),
        "forward_return_5d": rng.normal(0.005, 0.030, n),
        "mae_5d": rng.normal(-0.020, 0.010, n),
        "mfe_5d": rng.normal(0.025, 0.012, n),
    })


# ──────────────────────────────────────────────────────────────
# compute_decile_breakpoints
# ──────────────────────────────────────────────────────────────
class TestComputeDecileBreakpoints:
    def test_returns_9_breakpoints(self):
        panel = _make_panel(n=1000)
        bp = compute_decile_breakpoints(panel)
        assert len(bp) == 9

    def test_breakpoints_sorted_ascending(self):
        panel = _make_panel(n=1000)
        bp = compute_decile_breakpoints(panel)
        for i in range(len(bp) - 1):
            assert bp[i] <= bp[i + 1]

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            compute_decile_breakpoints(pd.DataFrame())

    def test_missing_column_raises(self):
        panel = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError):
            compute_decile_breakpoints(panel)


# ──────────────────────────────────────────────────────────────
# assign_deciles_to_panel
# ──────────────────────────────────────────────────────────────
class TestAssignDecilesToPanel:
    def test_assigns_to_full_decile_range(self):
        panel = _make_panel(n=1000)
        bp = compute_decile_breakpoints(panel)
        deciles = assign_deciles_to_panel(panel, bp)
        assert deciles.min() == 0
        assert deciles.max() == 9

    def test_approximately_uniform_distribution(self):
        panel = _make_panel(n=1000)
        bp = compute_decile_breakpoints(panel)
        deciles = assign_deciles_to_panel(panel, bp)
        counts = deciles.value_counts().sort_index()
        # Each decile should have roughly n/10 rows
        for c in counts.values:
            assert 50 < c < 200  # 100 ± 50 — generous


# ──────────────────────────────────────────────────────────────
# aggregate_bucket
# ──────────────────────────────────────────────────────────────
class TestAggregateBucket:
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "forward_return_5d": [0.01, 0.02, -0.01, 0.03, -0.02],
            "mae_5d": [-0.02, -0.01, -0.03, -0.005, -0.04],
            "mfe_5d": [0.03, 0.04, 0.01, 0.05, 0.02],
        })
        bucket = aggregate_bucket(df)
        assert bucket is not None
        assert bucket.n == 5
        assert bucket.mean_forward_return_5d == pytest.approx(0.006)
        assert bucket.win_rate_5d == pytest.approx(0.6)  # 3/5

    def test_empty_returns_none(self):
        assert aggregate_bucket(pd.DataFrame()) is None


# ──────────────────────────────────────────────────────────────
# build_calibration_table
# ──────────────────────────────────────────────────────────────
class TestBuildCalibrationTable:
    def test_produces_buckets_at_all_levels(self):
        panel = _make_panel(n=2000)
        table, bp = build_calibration_table(panel, min_bucket_n=10)

        assert ("global",) in table

        # Some decile buckets exist
        decile_keys = [k for k in table if len(k) == 1 and k != ("global",)]
        assert len(decile_keys) > 0

        # Some decile×regime buckets exist
        l2_keys = [k for k in table if len(k) == 2]
        assert len(l2_keys) > 0

        # Some level-3 buckets may exist (depends on sample)

    def test_min_bucket_n_filters_small_buckets(self):
        panel = _make_panel(n=200)
        table_high = build_calibration_table(panel, min_bucket_n=100)[0]
        table_low = build_calibration_table(panel, min_bucket_n=10)[0]
        # Higher min_n → fewer buckets
        assert len(table_high) < len(table_low)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            build_calibration_table(pd.DataFrame())

    def test_decile_bp_has_9_values(self):
        panel = _make_panel(n=500)
        _, bp = build_calibration_table(panel, min_bucket_n=10)
        assert len(bp) == 9


# ──────────────────────────────────────────────────────────────
# calibrate_panel end-to-end
# ──────────────────────────────────────────────────────────────
class TestCalibratePanel:
    def test_end_to_end(self, tmp_path):
        panel = _make_panel(n=1000)
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        cal_path = tmp_path / "cal.json"
        summary = calibrate_panel(
            panel_path=panel_path,
            output_path=cal_path,
            min_bucket_n=10,
        )

        assert cal_path.exists()
        assert summary["n_buckets"] > 0
        assert summary["n_panel_rows"] == 1000

        # Verify JSON is loadable
        table, metadata = load_calibration_table(cal_path)
        assert ("global",) in table
        assert "decile_breakpoints" in metadata
        assert len(metadata["decile_breakpoints"]) == 9

    def test_train_end_filters_panel(self, tmp_path):
        panel = _make_panel(n=1000)
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        cal_path = tmp_path / "cal.json"
        # Use mid-point as train_end
        mid = panel["date"].iloc[500]
        summary = calibrate_panel(
            panel_path=panel_path,
            output_path=cal_path,
            train_end=mid.strftime("%Y-%m-%d"),
            min_bucket_n=10,
        )
        # Train size should be ~500 (approx, depends on date inclusion)
        assert summary["n_train_rows"] < summary["n_panel_rows"]
