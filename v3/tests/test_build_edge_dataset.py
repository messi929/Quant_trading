"""V3.3 calibration panel builder tests (PR-1.4).

Verifies:
  - compute_forward_outcome (lookahead-safe, MAE/MFE/realized_vol)
  - assign_liquidity_bucket / assign_vol_state
  - verify_no_lookahead (uniqueness, identity-leak detection)
  - build_edge_dataset on synthetic OHLCV/macro/vol panel
  - panel_sanity_summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.research import (
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


# ──────────────────────────────────────────────────────────────
# Synthetic data generators
# ──────────────────────────────────────────────────────────────
def _make_ticker_ohlcv(
    ticker: str,
    start: str = "2026-01-01",
    n_days: int = 120,
    seed: int = 42,
    drift: float = 0.0005,
    vol: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    price = 100.0
    rows = []
    for d in dates:
        ret = float(rng.normal(drift, vol))
        price *= 1.0 + ret
        high = price * (1.0 + abs(float(rng.normal(0.0, 0.005))))
        low = price * (1.0 - abs(float(rng.normal(0.0, 0.005))))
        rows.append({
            "date": d,
            "ticker": ticker,
            "open": price,
            "high": max(price, high),
            "low": min(price, low),
            "close": price,
            "volume": 1_000_000 + int(rng.normal(0, 100_000)),
        })
    return pd.DataFrame(rows)


def _make_panel_ohlcv(n_tickers: int = 5, n_days: int = 120) -> pd.DataFrame:
    frames = [
        _make_ticker_ohlcv(f"T{i:02d}", n_days=n_days, seed=42 + i)
        for i in range(n_tickers)
    ]
    return pd.concat(frames, ignore_index=True)


def _make_macro_pctl(n_days: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "vix_ratio": rng.uniform(0.3, 0.7, n_days),
        "hy_level": rng.uniform(0.3, 0.6, n_days),
        "breadth": rng.uniform(0.4, 0.6, n_days),
        "yc_slope": rng.uniform(0.4, 0.6, n_days),
    }, index=dates)


def _make_vol_predictions(
    ohlcv: pd.DataFrame, vol_score_fn=None
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for date in ohlcv["date"].unique():
        for ticker in ohlcv["ticker"].unique():
            score = vol_score_fn(date, ticker) if vol_score_fn else float(rng.uniform(-0.5, 1.5))
            rows.append({
                "date": date,
                "ticker": ticker,
                "vol_score": score,
                "confidence": float(rng.uniform(0.3, 0.95)),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# compute_forward_outcome
# ──────────────────────────────────────────────────────────────
class TestComputeForwardOutcome:
    def test_basic_forward_returns(self):
        df = _make_ticker_ohlcv("AAPL", n_days=20, seed=1)
        t = df["date"].iloc[5]
        outcome = compute_forward_outcome(df, t)
        assert outcome is not None
        assert isinstance(outcome, ForwardOutcome)
        assert -0.5 < outcome.forward_return_5d < 0.5  # sanity range

    def test_returns_none_when_insufficient_forward(self):
        df = _make_ticker_ohlcv("AAPL", n_days=20, seed=1)
        # t is the last date — no t+10
        t = df["date"].iloc[-1]
        outcome = compute_forward_outcome(df, t)
        assert outcome is None

    def test_returns_none_for_empty(self):
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        t = pd.Timestamp("2026-01-15")
        outcome = compute_forward_outcome(empty, t)
        assert outcome is None

    def test_lookahead_safe_forward_returns_after_t(self):
        """forward_return must use prices > t, not include t itself."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=15),
            "open": [100.0] * 15,
            "high": [101.0] * 15,
            "low": [99.0] * 15,
            "close": [100.0, 100.0, 100.0, 100.0, 100.0,
                     110.0, 110.0, 110.0, 110.0, 110.0,
                     100.0, 100.0, 100.0, 100.0, 100.0],
        })
        # t at index 4 (close=100). Forward 1d (idx 5) is 110.
        t = df["date"].iloc[4]
        outcome = compute_forward_outcome(df, t)
        assert outcome is not None
        assert outcome.forward_return_1d == pytest.approx(0.10)  # 110/100 - 1
        assert outcome.forward_return_5d == pytest.approx(0.10)  # idx 9, 110

    def test_mae_mfe_within_5d_window(self):
        """MAE/MFE use t+1..t+5 only."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=15),
            "open": [100.0] * 15,
            "high": [105.0] * 15,
            "low": [95.0] * 15,
            "close": [100.0] * 15,
        })
        # Modify low at t+3 to test MAE
        df.loc[df.index[3], "low"] = 90.0  # before t+1, should NOT affect MAE
        df.loc[df.index[7], "low"] = 80.0  # at t+3 from anchor 4 — should affect

        t = df["date"].iloc[4]
        outcome = compute_forward_outcome(df, t)
        assert outcome is not None
        # MAE at t+3 = low 80 / close 100 - 1 = -0.20
        assert outcome.mae_5d == pytest.approx(-0.20)


# ──────────────────────────────────────────────────────────────
# Bucket helpers
# ──────────────────────────────────────────────────────────────
class TestBucketHelpers:
    def test_liquidity_bucket_low(self):
        assert assign_liquidity_bucket(50_000, (100_000, 1_000_000)) == "low"

    def test_liquidity_bucket_mid(self):
        assert assign_liquidity_bucket(500_000, (100_000, 1_000_000)) == "mid"

    def test_liquidity_bucket_high(self):
        assert assign_liquidity_bucket(2_000_000, (100_000, 1_000_000)) == "high"

    def test_vol_state_expanding(self):
        assert assign_vol_state(0.30, baseline_vol=0.20) == "expanding"

    def test_vol_state_contracting(self):
        assert assign_vol_state(0.10, baseline_vol=0.20) == "contracting"

    def test_vol_state_neutral(self):
        assert assign_vol_state(0.20, baseline_vol=0.20) == "neutral"

    def test_vol_state_zero_baseline_safe(self):
        assert assign_vol_state(0.20, baseline_vol=0.0) == "neutral"


# ──────────────────────────────────────────────────────────────
# verify_no_lookahead
# ──────────────────────────────────────────────────────────────
class TestVerifyNoLookahead:
    def test_empty_panel_ok(self):
        verify_no_lookahead(pd.DataFrame())  # no raise

    def test_unique_date_ticker_required(self):
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "ticker": ["AAPL", "AAPL"],  # duplicate
            "raw_opportunity": [0.01, 0.01],
            "direction": [0.05, 0.05],
            "conviction": [0.5, 0.5],
            "forward_return_5d": [0.02, 0.02],
        })
        with pytest.raises(AssertionError, match="Duplicate"):
            verify_no_lookahead(panel)

    def test_missing_forward_columns_raises(self):
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2026-01-01")],
            "ticker": ["AAPL"],
            "raw_opportunity": [0.01],
        })
        with pytest.raises(AssertionError, match="forward_return"):
            verify_no_lookahead(panel)

    def test_identity_leak_detected(self):
        # raw_opportunity == forward_return_5d in all rows → leak
        n = 5
        panel = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=n),
            "ticker": ["AAPL"] * n,
            "raw_opportunity": [0.02] * n,
            "direction": [0.05] * n,
            "conviction": [0.5] * n,
            "forward_return_5d": [0.02] * n,  # identical
            "forward_return_1d": [0.01] * n,
        })
        with pytest.raises(AssertionError, match="Identity leak"):
            verify_no_lookahead(panel)


# ──────────────────────────────────────────────────────────────
# build_edge_dataset (integration with synthetic data)
# ──────────────────────────────────────────────────────────────
class TestBuildEdgeDataset:
    @pytest.fixture
    def synthetic_inputs(self):
        ohlcv = _make_panel_ohlcv(n_tickers=10, n_days=180)
        macro = _make_macro_pctl(n_days=180)
        vol_pred = _make_vol_predictions(ohlcv)
        return ohlcv, macro, vol_pred

    def test_produces_panel_with_required_columns(self, synthetic_inputs, tmp_path):
        ohlcv, macro, vol_pred = synthetic_inputs
        cfg = BuildConfig(
            start="2026-01-01",
            end="2026-06-30",
            output_path=tmp_path / "panel.parquet",
            min_history=60,
        )
        panel = build_edge_dataset(
            cfg, ohlcv_data=ohlcv, macro_pctl=macro, vol_predictions=vol_pred,
        )
        assert not panel.empty
        # All required columns present
        for col in PANEL_COLUMNS:
            assert col in panel.columns
        # Saved
        assert (tmp_path / "panel.parquet").exists()

    def test_panel_passes_lookahead_check(self, synthetic_inputs):
        ohlcv, macro, vol_pred = synthetic_inputs
        cfg = BuildConfig(start="2026-01-01", end="2026-06-30", min_history=60)
        panel = build_edge_dataset(cfg, ohlcv, macro, vol_pred)
        # build_edge_dataset calls verify_no_lookahead internally.
        # If panel returned, verification passed.
        assert len(panel) > 0
        # Re-verify here for assurance
        verify_no_lookahead(panel)

    def test_panel_unique_date_ticker(self, synthetic_inputs):
        ohlcv, macro, vol_pred = synthetic_inputs
        cfg = BuildConfig(start="2026-01-01", end="2026-06-30", min_history=60)
        panel = build_edge_dataset(cfg, ohlcv, macro, vol_pred)
        n_unique = panel.groupby(["date", "ticker"]).size().shape[0]
        assert len(panel) == n_unique

    def test_forward_returns_within_realistic_range(self, synthetic_inputs):
        ohlcv, macro, vol_pred = synthetic_inputs
        cfg = BuildConfig(start="2026-01-01", end="2026-06-30", min_history=60)
        panel = build_edge_dataset(cfg, ohlcv, macro, vol_pred)
        # Synthetic returns ~N(0, 0.02) per day → 5d cumulative ~N(0, 0.045)
        # Outliers possible but mean should be near 0
        mean_5d = panel["forward_return_5d"].mean()
        assert abs(mean_5d) < 0.05, f"mean forward 5d return suspicious: {mean_5d}"

    def test_empty_period_returns_empty_panel(self, synthetic_inputs):
        ohlcv, macro, vol_pred = synthetic_inputs
        # Period before any data
        cfg = BuildConfig(start="2020-01-01", end="2020-06-30", min_history=60)
        panel = build_edge_dataset(cfg, ohlcv, macro, vol_pred)
        assert panel.empty
        assert list(panel.columns) == PANEL_COLUMNS


# ──────────────────────────────────────────────────────────────
# panel_sanity_summary
# ──────────────────────────────────────────────────────────────
class TestPanelSanitySummary:
    def test_summary_empty(self):
        s = panel_sanity_summary(pd.DataFrame())
        assert s == {"n_rows": 0, "n_tickers": 0, "n_dates": 0}

    def test_summary_nonempty(self):
        n = 10
        panel = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=n),
            "ticker": [f"T{i:02d}" for i in range(n)],
            "raw_opportunity": np.linspace(-0.05, 0.05, n),
            "direction": np.linspace(-0.05, 0.05, n),
            "conviction": np.linspace(0.0, 1.0, n),
            "regime": ["neutral"] * n,
            "regime_score": [0.5] * n,
            "sector": ["tech"] * n,
            "liquidity_bucket": ["high"] * n,
            "vol_state": ["neutral"] * n,
            "forward_return_1d": np.zeros(n),
            "forward_return_3d": np.zeros(n),
            "forward_return_5d": np.linspace(-0.02, 0.02, n),
            "forward_return_10d": np.zeros(n),
            "mae_5d": np.zeros(n),
            "mfe_5d": np.zeros(n),
            "realized_vol_5d": np.full(n, 0.20),
        })
        s = panel_sanity_summary(panel)
        assert s["n_rows"] == n
        assert s["regime_counts"] == {"neutral": n}
        assert "missing_pct" in s
