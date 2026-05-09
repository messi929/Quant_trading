"""V3.3 EdgeTierSystem tests (PR-2.3).

Verifies:
  - DecisionPolicy protocol satisfied
  - TierThresholds save/load round-trip
  - derive_tier_thresholds from panel distribution
  - assign() S/A/B/C/BLOCKED classification
  - S-tier demotion when win_prob/mae insufficient
  - tier_rank ordering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.strategy._base import DecisionPolicy
from v3.strategy.edge_tier import (
    EdgeTierSystem,
    TIER_RANK,
    TierThresholds,
    derive_tier_thresholds,
    load_tier_thresholds,
    save_tier_thresholds,
    tier_rank,
)
from v3.strategy.types import EdgeCandidate


def _candidate(
    net_edge: float | None = 0.005,
    win_prob: float = 0.55,
    mae: float = -0.025,
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker="AAPL",
        direction=0.05,
        conviction=0.7,
        raw_opportunity=0.005,
        regime="neutral",
        regime_score=0.5,
        liquidity_bucket="high",
        expected_return_5d=0.010,
        expected_mae_5d=mae,
        expected_mfe_5d=0.025,
        win_probability_5d=win_prob,
        calibration_n=200,
        net_edge_5d=net_edge,
    )


# ──────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────
class TestProtocolConformance:
    def test_implements_decision_policy(self):
        sys = EdgeTierSystem(thresholds=TierThresholds())
        assert isinstance(sys, DecisionPolicy)

    def test_name_attribute(self):
        sys = EdgeTierSystem(thresholds=TierThresholds())
        assert sys.name == "edge_tier"

    def test_enabled_property(self):
        sys = EdgeTierSystem(thresholds=TierThresholds(), enabled=True)
        assert sys.enabled is True
        sys2 = EdgeTierSystem(thresholds=TierThresholds(), enabled=False)
        assert sys2.enabled is False


# ──────────────────────────────────────────────────────────────
# TierThresholds I/O
# ──────────────────────────────────────────────────────────────
class TestThresholdIO:
    def test_save_load_round_trip(self, tmp_path):
        original = TierThresholds(
            s_tier_net_edge=0.012,
            a_tier_net_edge=0.005,
            b_tier_net_edge=-0.001,
            s_tier_winprob=0.65,
            a_tier_winprob=0.50,
            s_tier_max_mae=-0.030,
        )
        path = tmp_path / "thresholds.json"
        save_tier_thresholds(original, path, metadata={"version": "test"})

        loaded, metadata = load_tier_thresholds(path)
        assert loaded == original
        assert metadata["version"] == "test"

    def test_from_file(self, tmp_path):
        thresholds = TierThresholds(s_tier_net_edge=0.015)
        path = tmp_path / "thresholds.json"
        save_tier_thresholds(thresholds, path)

        sys = EdgeTierSystem.from_file(path)
        assert sys.thresholds.s_tier_net_edge == pytest.approx(0.015)


# ──────────────────────────────────────────────────────────────
# Threshold derivation
# ──────────────────────────────────────────────────────────────
class TestDeriveThresholds:
    def test_basic_derivation(self):
        # net_edge ~ N(0.001, 0.005), 1000 samples
        rng = np.random.default_rng(42)
        net_edges = rng.normal(0.001, 0.005, 1000)
        thresholds = derive_tier_thresholds(net_edges)
        # Sanity: s > a > b
        assert thresholds.s_tier_net_edge > thresholds.a_tier_net_edge
        assert thresholds.a_tier_net_edge > thresholds.b_tier_net_edge

    def test_with_win_probs(self):
        rng = np.random.default_rng(42)
        net_edges = rng.normal(0.001, 0.005, 500)
        win_probs = rng.uniform(0.40, 0.70, 500)
        t = derive_tier_thresholds(net_edges, win_probs=win_probs)
        # s_tier_winprob from 70th pct of [0.40, 0.70]
        assert 0.40 < t.s_tier_winprob < 0.70

    def test_with_maes(self):
        rng = np.random.default_rng(42)
        net_edges = rng.normal(0.001, 0.005, 500)
        maes = rng.uniform(-0.05, -0.005, 500)
        t = derive_tier_thresholds(net_edges, expected_maes=maes)
        # s_tier_max_mae should be ~ 30th best (least negative) in distribution
        # i.e., upper end of the maes
        assert -0.05 < t.s_tier_max_mae < -0.005

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            derive_tier_thresholds([])


# ──────────────────────────────────────────────────────────────
# assign() — tier classification
# ──────────────────────────────────────────────────────────────
class TestAssign:
    def _sys(self, **kwargs) -> EdgeTierSystem:
        # Default thresholds
        defaults = dict(
            s_tier_net_edge=0.0090,
            a_tier_net_edge=0.0040,
            b_tier_net_edge=0.0000,
            s_tier_winprob=0.62,
            a_tier_winprob=0.55,
            s_tier_max_mae=-0.025,
        )
        defaults.update(kwargs)
        return EdgeTierSystem(thresholds=TierThresholds(**defaults))

    def test_s_tier_all_pass(self):
        sys = self._sys()
        c = _candidate(net_edge=0.012, win_prob=0.70, mae=-0.020)
        assert sys.assign(c).edge_tier == "S"

    def test_s_tier_demotes_on_low_winprob(self):
        sys = self._sys()
        # net_edge passes S, but win_prob fails → A
        c = _candidate(net_edge=0.012, win_prob=0.55, mae=-0.020)
        assert sys.assign(c).edge_tier == "A"

    def test_s_tier_demotes_on_bad_mae(self):
        sys = self._sys()
        # net_edge passes S, win_prob passes, but mae too negative → A
        c = _candidate(net_edge=0.012, win_prob=0.70, mae=-0.050)
        assert sys.assign(c).edge_tier == "A"

    def test_a_tier_normal(self):
        sys = self._sys()
        c = _candidate(net_edge=0.005, win_prob=0.60, mae=-0.030)
        assert sys.assign(c).edge_tier == "A"

    def test_a_tier_fails_when_winprob_low(self):
        sys = self._sys()
        # Above A net_edge but below A winprob → drops to B
        c = _candidate(net_edge=0.005, win_prob=0.40, mae=-0.030)
        assert sys.assign(c).edge_tier == "B"

    def test_b_tier(self):
        sys = self._sys()
        c = _candidate(net_edge=0.001, win_prob=0.55, mae=-0.030)
        assert sys.assign(c).edge_tier == "B"

    def test_c_tier_below_zero(self):
        sys = self._sys()
        c = _candidate(net_edge=-0.005, win_prob=0.55, mae=-0.030)
        assert sys.assign(c).edge_tier == "C"

    def test_blocked_no_net_edge(self):
        sys = self._sys()
        c = _candidate(net_edge=None, win_prob=0.55, mae=-0.030)
        assert sys.assign(c).edge_tier == "BLOCKED"

    def test_assign_returns_new_instance(self):
        sys = self._sys()
        c = _candidate()
        result = sys.assign(c)
        assert c.edge_tier is None  # original unchanged
        assert result.edge_tier is not None
        assert result is not c


# ──────────────────────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────────────────────
class TestAssignBatch:
    def test_batch_returns_list(self):
        sys = EdgeTierSystem(thresholds=TierThresholds())
        cands = [
            _candidate(net_edge=0.012, win_prob=0.70, mae=-0.020),  # S
            _candidate(net_edge=0.005, win_prob=0.60, mae=-0.030),  # A
            _candidate(net_edge=0.001, win_prob=0.55, mae=-0.030),  # B
            _candidate(net_edge=-0.005, win_prob=0.55, mae=-0.030), # C
        ]
        result = sys.assign_batch(cands)
        tiers = [c.edge_tier for c in result]
        assert tiers == ["S", "A", "B", "C"]


# ──────────────────────────────────────────────────────────────
# tier_rank
# ──────────────────────────────────────────────────────────────
class TestTierRank:
    def test_ordering(self):
        assert tier_rank("S") > tier_rank("A")
        assert tier_rank("A") > tier_rank("B")
        assert tier_rank("B") > tier_rank("C")
        assert tier_rank("C") > tier_rank("BLOCKED")

    def test_none_safe(self):
        assert tier_rank(None) == -1

    def test_unknown_tier_safe(self):
        assert tier_rank("XYZ") == -1
