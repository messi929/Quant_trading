"""V3.3 validate_edge.py tests (PR-2.1).

Verifies:
  - validate_oos with synthetic panel (PASS / FAIL detection)
  - compute_decile_stats
  - render_markdown_report (smoke)
  - decile monotonicity Spearman
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.research.calibrate_edge import calibrate_panel
from v3.research.validate_edge import (
    DecileStats,
    ValidationResult,
    compute_decile_stats,
    render_markdown_report,
    validate_oos,
)


def _make_monotonic_panel(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic panel where forward_return_5d correlates with raw_opportunity.

    This should produce strong decile monotonicity in OOS validation.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.002, 0.005, n)
    # forward_return is correlated with raw + noise
    fwd = 0.5 * raw + rng.normal(0, 0.020, n)
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "ticker": [f"T{i:04d}" for i in range(n)],
        "raw_opportunity": raw,
        "regime": rng.choice(["neutral", "bull"], size=n, p=[0.7, 0.3]),
        "liquidity_bucket": rng.choice(["high", "mid"], size=n, p=[0.6, 0.4]),
        "forward_return_5d": fwd,
        "mae_5d": rng.normal(-0.020, 0.010, n),
        "mfe_5d": rng.normal(0.025, 0.012, n),
    })


def _make_random_panel(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Panel with NO correlation between raw_opportunity and forward_return."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "ticker": [f"T{i:04d}" for i in range(n)],
        "raw_opportunity": rng.normal(0.002, 0.005, n),
        "regime": rng.choice(["neutral", "bull"], size=n, p=[0.7, 0.3]),
        "liquidity_bucket": rng.choice(["high", "mid"], size=n, p=[0.6, 0.4]),
        "forward_return_5d": rng.normal(0.001, 0.020, n),  # independent
        "mae_5d": rng.normal(-0.020, 0.010, n),
        "mfe_5d": rng.normal(0.025, 0.012, n),
    })


# ──────────────────────────────────────────────────────────────
# DecileStats
# ──────────────────────────────────────────────────────────────
class TestComputeDecileStats:
    def test_single_decile(self):
        panel = pd.DataFrame({
            "_decile": [3, 3, 3],
            "forward_return_5d": [0.01, 0.02, -0.01],
            "expected_return_5d": [0.005, 0.005, 0.005],
        })
        stats = compute_decile_stats(panel)
        assert len(stats) == 1
        s = stats[0]
        assert s.decile == 3
        assert s.n == 3
        assert s.mean_fwd_5d == pytest.approx(0.00666, rel=1e-2)
        assert s.win_rate_5d == pytest.approx(2 / 3, rel=1e-3)

    def test_multiple_deciles_sorted(self):
        panel = pd.DataFrame({
            "_decile": [1, 1, 5, 5, 9],
            "forward_return_5d": [0.01, 0.02, 0.03, 0.04, 0.05],
            "expected_return_5d": [0.001, 0.001, 0.005, 0.005, 0.009],
        })
        stats = compute_decile_stats(panel)
        deciles = [s.decile for s in stats]
        assert deciles == sorted(deciles)


# ──────────────────────────────────────────────────────────────
# validate_oos
# ──────────────────────────────────────────────────────────────
class TestValidateOos:
    def test_monotonic_panel_passes(self, tmp_path):
        # Build calibration on first half, validate on second
        panel = _make_monotonic_panel(n=2000)
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        cal_path = tmp_path / "cal.json"
        train_end = panel["date"].iloc[1000].strftime("%Y-%m-%d")
        calibrate_panel(
            panel_path=panel_path,
            output_path=cal_path,
            train_end=train_end,
            min_bucket_n=10,
        )

        result = validate_oos(
            panel_path=panel_path,
            calibration_path=cal_path,
            train_end=train_end,
        )
        # Synthetic correlation should yield positive monotonicity
        assert result.decile_monotonicity_spearman > 0.5
        assert result.top_decile_mean_positive
        assert result.passed

    def test_random_panel_fails_monotonicity(self, tmp_path):
        panel = _make_random_panel(n=2000)
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        cal_path = tmp_path / "cal.json"
        train_end = panel["date"].iloc[1000].strftime("%Y-%m-%d")
        calibrate_panel(
            panel_path=panel_path,
            output_path=cal_path,
            train_end=train_end,
            min_bucket_n=10,
        )

        result = validate_oos(
            panel_path=panel_path,
            calibration_path=cal_path,
            train_end=train_end,
        )
        # Random panel should NOT pass strong monotonicity
        assert abs(result.decile_monotonicity_spearman) < 0.5

    def test_empty_oos_returns_zero_result(self, tmp_path):
        panel = _make_monotonic_panel(n=500)
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        cal_path = tmp_path / "cal.json"
        # train_end after panel — OOS empty
        future = "2030-01-01"
        calibrate_panel(
            panel_path=panel_path,
            output_path=cal_path,
            min_bucket_n=10,
        )

        result = validate_oos(
            panel_path=panel_path,
            calibration_path=cal_path,
            train_end=future,
        )
        assert result.overall_n == 0
        assert not result.passed


# ──────────────────────────────────────────────────────────────
# render_markdown_report
# ──────────────────────────────────────────────────────────────
class TestRenderMarkdownReport:
    def test_smoke(self):
        result = ValidationResult(
            decile_stats=[
                DecileStats(
                    decile=0, n=100, mean_fwd_5d=-0.005, median_fwd_5d=-0.004,
                    win_rate_5d=0.40, profit_factor=0.7, mean_calibrated=-0.005,
                ),
                DecileStats(
                    decile=9, n=100, mean_fwd_5d=0.015, median_fwd_5d=0.012,
                    win_rate_5d=0.65, profit_factor=2.5, mean_calibrated=0.015,
                ),
            ],
            decile_monotonicity_spearman=0.85,
            top_minus_bottom_decile_mean=0.020,
            top_decile_mean_positive=True,
            overall_n=200,
            train_end="2025-12-31",
        )
        md = render_markdown_report(result)
        assert "PASS" in md
        assert "0.020" in md  # top - bottom value (4-decimal format)
        assert "0.850" in md  # monotonicity (3-decimal format)
        assert "decile" in md.lower()

    def test_failure_status_renders(self):
        result = ValidationResult(
            decile_stats=[],
            decile_monotonicity_spearman=0.1,
            top_minus_bottom_decile_mean=0.001,
            top_decile_mean_positive=False,
            overall_n=100,
            train_end="2025-12-31",
        )
        md = render_markdown_report(result)
        assert "FAIL" in md
