"""V3.3 EdgeEngine tests (PR-2.2).

Verifies:
  - CostModel decomposition (expected_cost + slippage_buffer)
  - EdgePolicy default values
  - EdgeEngine.compute monotonicity:
      higher expected_return → higher net_edge
      higher cost → lower net_edge
      higher |expected_mae| → lower net_edge
  - calibration_n shrinkage when n < demote threshold
  - threshold gate: entry_pass when net_edge > threshold
  - derive_entry_threshold from percentile
  - not_calibrated rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.strategy.edge_engine import (
    CostModel,
    EdgeEngine,
    EdgePolicy,
    derive_entry_threshold,
)
from v3.strategy.types import EdgeCandidate


def _calibrated_candidate(
    expected_return_5d: float = 0.010,
    expected_mae_5d: float = -0.020,
    expected_mfe_5d: float = 0.025,
    win_probability_5d: float = 0.55,
    calibration_n: int = 200,
    raw: float = 0.005,
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker="AAPL",
        direction=0.05,
        conviction=0.7,
        raw_opportunity=raw,
        regime="neutral",
        regime_score=0.5,
        liquidity_bucket="high",
        expected_return_5d=expected_return_5d,
        expected_mae_5d=expected_mae_5d,
        expected_mfe_5d=expected_mfe_5d,
        win_probability_5d=win_probability_5d,
        calibration_n=calibration_n,
        calibration_key=(5, "neutral"),
    )


# ──────────────────────────────────────────────────────────────
# CostModel
# ──────────────────────────────────────────────────────────────
class TestCostModel:
    def test_expected_cost_default(self):
        cm = CostModel()
        # 0.0002 + 0.0003 + 0.0005 = 0.0010
        assert cm.expected_cost() == pytest.approx(0.0010)

    def test_slippage_buffer_min_clip(self):
        cm = CostModel()
        # vol=0 → buf = 0.0010 + 0 = 0.0010 (== min)
        assert cm.slippage_buffer("AAPL", 0.0) == pytest.approx(0.0010)

    def test_slippage_buffer_max_clip(self):
        cm = CostModel()
        # vol=10 (extreme) → buf = 0.0010 + 1.0 → clipped to 0.0040
        assert cm.slippage_buffer("AAPL", 10.0) == pytest.approx(0.0040)

    def test_slippage_buffer_scales_with_vol(self):
        cm = CostModel()
        # multiplier=0.10. With min/max clip [0.0010, 0.0040], pre-clip range
        # of vol where it matters: (0.0010 - 0.0010)/0.10 = 0 ~ 0.030
        low = cm.slippage_buffer("AAPL", 0.005)   # 0.0010 + 0.0005 = 0.0015
        high = cm.slippage_buffer("AAPL", 0.020)  # 0.0010 + 0.0020 = 0.0030
        assert high > low


# ──────────────────────────────────────────────────────────────
# EdgePolicy
# ──────────────────────────────────────────────────────────────
class TestEdgePolicy:
    def test_defaults(self):
        p = EdgePolicy()
        assert p.entry_threshold == pytest.approx(0.0040)
        assert p.lambda_mae == pytest.approx(0.20)
        assert p.calibration_n_demote == 100


# ──────────────────────────────────────────────────────────────
# EdgeEngine.compute — monotonicity
# ──────────────────────────────────────────────────────────────
class TestComputeMonotonicity:
    def test_higher_expected_return_higher_net_edge(self):
        eng = EdgeEngine()
        low = eng.compute(_calibrated_candidate(expected_return_5d=0.005))
        high = eng.compute(_calibrated_candidate(expected_return_5d=0.020))
        assert high.net_edge_5d > low.net_edge_5d

    def test_worse_mae_lower_net_edge(self):
        eng = EdgeEngine()
        low_mae = eng.compute(_calibrated_candidate(expected_mae_5d=-0.010))
        high_mae = eng.compute(_calibrated_candidate(expected_mae_5d=-0.040))
        # high_mae has bigger |mae| → bigger penalty → lower net_edge
        assert low_mae.net_edge_5d > high_mae.net_edge_5d

    def test_higher_cost_lower_net_edge(self):
        cheap = EdgeEngine(cost_model=CostModel(commission_roundtrip=0.0001))
        expensive = EdgeEngine(cost_model=CostModel(commission_roundtrip=0.0010))
        c = _calibrated_candidate()
        cheap_result = cheap.compute(c)
        expensive_result = expensive.compute(c)
        assert cheap_result.net_edge_5d > expensive_result.net_edge_5d


# ──────────────────────────────────────────────────────────────
# EdgeEngine.compute — fields populated
# ──────────────────────────────────────────────────────────────
class TestComputePopulatesFields:
    def test_all_fields_populated(self):
        eng = EdgeEngine()
        result = eng.compute(_calibrated_candidate())
        assert result.expected_cost is not None
        assert result.slippage_buffer is not None
        assert result.downside_risk_penalty is not None
        assert result.net_edge_5d is not None

    def test_immutable_returns_new_instance(self):
        eng = EdgeEngine()
        c = _calibrated_candidate()
        result = eng.compute(c)
        assert result is not c
        assert c.net_edge_5d is None  # original unchanged

    def test_lambda_mae_applied_correctly(self):
        eng = EdgeEngine(policy=EdgePolicy(lambda_mae=0.50))
        # mae=-0.030 → penalty = 0.50 × 0.030 = 0.015
        result = eng.compute(_calibrated_candidate(
            expected_return_5d=0.020,
            expected_mae_5d=-0.030,
        ))
        # net = 0.020 - 0.001 - slip - 0.015
        # cost=0.0010, slip @ vol=0.20 = 0.0010 + 0.020 = 0.0030 (clipped)
        # net = 0.020 - 0.0010 - 0.0030 - 0.015 = 0.0010
        assert result.downside_risk_penalty == pytest.approx(0.015)


# ──────────────────────────────────────────────────────────────
# Calibration_n shrinkage
# ──────────────────────────────────────────────────────────────
class TestCalibrationShrinkage:
    def test_low_n_applies_demote(self):
        eng = EdgeEngine(policy=EdgePolicy(
            calibration_n_demote=100, demote_factor=0.5
        ))
        c_high = _calibrated_candidate(calibration_n=500)
        c_low = _calibrated_candidate(calibration_n=50)
        r_high = eng.compute(c_high)
        r_low = eng.compute(c_low)
        # Low n → shrunk → smaller net_edge
        assert r_high.net_edge_5d > r_low.net_edge_5d
        assert r_low.net_edge_5d == pytest.approx(r_high.net_edge_5d * 0.5, rel=1e-6)

    def test_high_n_no_shrinkage(self):
        eng = EdgeEngine(policy=EdgePolicy(
            calibration_n_demote=100, demote_factor=0.5
        ))
        c = _calibrated_candidate(calibration_n=200)
        r = eng.compute(c)  # default ticker_vol=0.20 → slip clipped to max 0.004
        # No shrinkage: net = expected_return - cost - slip(max-clipped) - mae_penalty
        # 0.010 - 0.001 - 0.004 - 0.004 = 0.001
        expected = 0.010 - 0.001 - 0.004 - 0.004
        assert r.net_edge_5d == pytest.approx(expected, rel=1e-6)


# ──────────────────────────────────────────────────────────────
# entry_pass gate
# ──────────────────────────────────────────────────────────────
class TestEntryPassGate:
    def test_high_net_edge_passes(self):
        eng = EdgeEngine(policy=EdgePolicy(entry_threshold=0.001))
        # net_edge = 0.020 - 0.001 - 0.003 - 0.004 = 0.012 > 0.001
        result = eng.compute(_calibrated_candidate(expected_return_5d=0.020))
        assert result.entry_pass

    def test_low_net_edge_fails(self):
        eng = EdgeEngine(policy=EdgePolicy(entry_threshold=0.010))
        # net_edge ~= 0.002 < 0.010
        result = eng.compute(_calibrated_candidate(expected_return_5d=0.010))
        assert not result.entry_pass
        assert result.reject_reason == "net_edge_below_threshold"


# ──────────────────────────────────────────────────────────────
# Not-calibrated rejection
# ──────────────────────────────────────────────────────────────
class TestNotCalibrated:
    def test_rejects_when_no_expected_return(self):
        eng = EdgeEngine()
        c = EdgeCandidate(
            date=pd.Timestamp("2026-05-09"),
            ticker="AAPL",
            direction=0.05,
            conviction=0.7,
            raw_opportunity=0.005,
            regime="neutral",
            regime_score=0.5,
            # expected_return_5d not set
        )
        result = eng.compute(c)
        assert not result.entry_pass
        assert result.reject_reason == "not_calibrated"


# ──────────────────────────────────────────────────────────────
# derive_entry_threshold
# ──────────────────────────────────────────────────────────────
class TestDeriveEntryThreshold:
    def test_70th_percentile(self):
        edges = list(np.linspace(-0.01, 0.02, 100))
        thresh = derive_entry_threshold(edges, percentile=0.70)
        # 70th percentile of linspace(-0.01, 0.02, 100)
        # ≈ -0.01 + 0.7 × 0.03 = 0.011
        assert thresh == pytest.approx(0.011, rel=1e-2)

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            derive_entry_threshold([0.01, 0.02], percentile=1.5)
        with pytest.raises(ValueError):
            derive_entry_threshold([0.01, 0.02], percentile=0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            derive_entry_threshold([], percentile=0.7)


# ──────────────────────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────────────────────
class TestComputeBatch:
    def test_batch_returns_list(self):
        eng = EdgeEngine()
        cands = [
            _calibrated_candidate(expected_return_5d=0.005),
            _calibrated_candidate(expected_return_5d=0.020),
        ]
        result = eng.compute_batch(cands)
        assert len(result) == 2
        assert result[0].net_edge_5d < result[1].net_edge_5d

    def test_batch_uses_per_ticker_vol(self):
        eng = EdgeEngine()
        cands = [_calibrated_candidate()]
        cands[0] = cands[0].with_calibration()  # ensure new instance
        result = eng.compute_batch(cands, ticker_vols={"AAPL": 0.50})
        # vol=0.50 → slip = 0.001 + 0.05 → clip to 0.004
        assert result[0].slippage_buffer == pytest.approx(0.004)
