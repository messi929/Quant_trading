"""V3.3 EdgeCalibrator tests (PR-2.1).

Verifies:
  - serialize / deserialize bucket key (round-trip)
  - save / load calibration table (round-trip)
  - assign_decile (boundary cases)
  - calibrate fallback chain (Level 1 → 2 → 3 → global)
  - insufficient_action policies (block, global, demote)
"""

from __future__ import annotations

import pandas as pd
import pytest

from v3.strategy.edge_calibrator import (
    CalibrationConfig,
    EdgeCalibrator,
    assign_decile,
    deserialize_bucket_key,
    load_calibration_table,
    save_calibration_table,
    serialize_bucket_key,
)
from v3.strategy.types import CalibrationBucket, EdgeCandidate


def _make_bucket(key: tuple, n: int = 100, mean: float = 0.005) -> CalibrationBucket:
    return CalibrationBucket(
        key=key,
        n=n,
        mean_forward_return_5d=mean,
        mean_mae_5d=-0.020,
        mean_mfe_5d=0.025,
        win_rate_5d=0.55,
        std_forward_return_5d=0.030,
        median_forward_return_5d=mean * 0.9,
    )


def _make_candidate(
    raw: float = 0.005,
    regime: str = "neutral",
    liquidity: str | None = "high",
) -> EdgeCandidate:
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-09"),
        ticker="AAPL",
        direction=0.05,
        conviction=raw / 0.05 if raw != 0 else 0.5,
        raw_opportunity=raw,
        regime=regime,
        regime_score=0.5,
        liquidity_bucket=liquidity,
    )


# ──────────────────────────────────────────────────────────────
# Key serialization
# ──────────────────────────────────────────────────────────────
class TestKeySerialization:
    def test_round_trip_global(self):
        key = ("global",)
        s = serialize_bucket_key(key)
        assert deserialize_bucket_key(s) == key

    def test_round_trip_decile(self):
        key = (5,)
        s = serialize_bucket_key(key)
        # Note: int → str on serialize, stays str on deserialize (acceptable)
        assert deserialize_bucket_key(s) == ("5",)

    def test_round_trip_with_none(self):
        key = (5, "neutral", None)
        s = serialize_bucket_key(key)
        assert deserialize_bucket_key(s) == ("5", "neutral", None)


# ──────────────────────────────────────────────────────────────
# Save / load round-trip
# ──────────────────────────────────────────────────────────────
class TestSaveLoadRoundTrip:
    def test_basic_round_trip(self, tmp_path):
        table = {
            ("global",): _make_bucket(("global",), n=1000, mean=0.003),
            (5,): _make_bucket((5,), n=500, mean=0.005),
            (5, "neutral"): _make_bucket((5, "neutral"), n=200, mean=0.006),
        }
        metadata = {
            "decile_breakpoints": [-0.01, -0.005, -0.001, 0.0,
                                   0.001, 0.003, 0.005, 0.008, 0.012],
            "version": "test",
        }
        path = tmp_path / "calibration.json"
        save_calibration_table(table, path, metadata=metadata)

        loaded, loaded_meta = load_calibration_table(path)
        assert len(loaded) == len(table)
        assert loaded_meta["version"] == "test"
        # Spot-check
        gb = loaded[("global",)]
        assert gb.n == 1000
        assert gb.mean_forward_return_5d == pytest.approx(0.003)


# ──────────────────────────────────────────────────────────────
# Decile assignment
# ──────────────────────────────────────────────────────────────
class TestAssignDecile:
    def test_below_first_breakpoint(self):
        bp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert assign_decile(0.05, bp) == 0

    def test_above_last_breakpoint(self):
        bp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert assign_decile(0.95, bp) == 9

    def test_middle_value(self):
        bp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # value < 0.5 means decile 4 (since bp[4] = 0.5, value 0.45 falls into idx 4)
        assert assign_decile(0.45, bp) == 4

    def test_exact_breakpoint_goes_to_higher_decile(self):
        # value < bp[i] check — value == bp[i] does NOT match → falls to next
        bp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert assign_decile(0.5, bp) == 5

    def test_invalid_breakpoints_raises(self):
        with pytest.raises(ValueError):
            assign_decile(0.5, [0.1, 0.2])  # too few


# ──────────────────────────────────────────────────────────────
# Calibrate — fallback chain
# ──────────────────────────────────────────────────────────────
class TestCalibrateFallbackChain:
    def _bp(self) -> list[float]:
        return [-0.01, -0.005, -0.001, 0.0, 0.001, 0.003, 0.005, 0.008, 0.012]

    def test_uses_full_key_when_available(self):
        # decile=5 (raw=0.004 maps to decile 5 since 0.003 ≤ 0.004 < 0.005)
        # Actually 0.004 < 0.005 → decile 5? Let me check:
        # bp = [-0.01, -0.005, -0.001, 0.0, 0.001, 0.003, 0.005, 0.008, 0.012]
        # 0.004 < 0.005 (idx 6) → decile 6
        c = _make_candidate(raw=0.004, regime="neutral", liquidity="high")
        full_key = (6, "neutral", "high")
        cal = EdgeCalibrator(
            table={
                full_key: _make_bucket(full_key, n=100, mean=0.010),
                (6, "neutral"): _make_bucket((6, "neutral"), n=200, mean=0.005),
                (6,): _make_bucket((6,), n=500, mean=0.003),
            },
            decile_breakpoints=self._bp(),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.010)  # full key
        assert result.calibration_key == full_key
        assert result.calibration_n == 100

    def test_falls_back_to_decile_regime(self):
        c = _make_candidate(raw=0.004, regime="neutral", liquidity="high")
        cal = EdgeCalibrator(
            table={
                # No full key, only level 2
                (6, "neutral"): _make_bucket((6, "neutral"), n=100, mean=0.005),
                (6,): _make_bucket((6,), n=500, mean=0.003),
            },
            decile_breakpoints=self._bp(),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.005)
        assert result.calibration_key == (6, "neutral")

    def test_falls_back_to_decile_only(self):
        c = _make_candidate(raw=0.004, regime="neutral", liquidity="high")
        cal = EdgeCalibrator(
            table={(6,): _make_bucket((6,), n=500, mean=0.003)},
            decile_breakpoints=self._bp(),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.003)
        assert result.calibration_key == (6,)

    def test_falls_back_to_global(self):
        c = _make_candidate(raw=0.004, regime="neutral", liquidity="high")
        cal = EdgeCalibrator(
            table={("global",): _make_bucket(("global",), n=10000, mean=0.001)},
            decile_breakpoints=self._bp(),
            config=CalibrationConfig(insufficient_action="global"),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.001)
        assert result.calibration_key == ("global",)


# ──────────────────────────────────────────────────────────────
# Insufficient action policies
# ──────────────────────────────────────────────────────────────
class TestInsufficientAction:
    def _bp(self) -> list[float]:
        return [-0.01, -0.005, -0.001, 0.0, 0.001, 0.003, 0.005, 0.008, 0.012]

    def test_block_policy_rejects(self):
        c = _make_candidate(raw=0.004)
        cal = EdgeCalibrator(
            table={},  # empty
            decile_breakpoints=self._bp(),
            config=CalibrationConfig(insufficient_action="block"),
        )
        result = cal.calibrate(c)
        assert not result.entry_pass
        assert result.reject_reason == "insufficient_calibration"
        assert result.expected_return_5d is None

    def test_demote_policy_shrinks(self):
        c = _make_candidate(raw=0.004)
        cal = EdgeCalibrator(
            table={("global",): _make_bucket(("global",), n=10000, mean=0.010)},
            decile_breakpoints=self._bp(),
            config=CalibrationConfig(
                insufficient_action="demote", demote_factor=0.5
            ),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.005)  # 0.010 × 0.5

    def test_global_policy_no_shrinkage(self):
        c = _make_candidate(raw=0.004)
        cal = EdgeCalibrator(
            table={("global",): _make_bucket(("global",), n=10000, mean=0.010)},
            decile_breakpoints=self._bp(),
            config=CalibrationConfig(insufficient_action="global"),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.010)

    def test_below_min_n_treated_as_missing(self):
        c = _make_candidate(raw=0.004, regime="neutral", liquidity="high")
        cal = EdgeCalibrator(
            table={
                # Full key has n=10 (below min 30) → fallback
                (6, "neutral", "high"): _make_bucket(
                    (6, "neutral", "high"), n=10, mean=0.999
                ),
                (6,): _make_bucket((6,), n=500, mean=0.003),
            },
            decile_breakpoints=self._bp(),
            config=CalibrationConfig(min_bucket_n=30),
        )
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.003)


# ──────────────────────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────────────────────
class TestCalibrateBatch:
    def test_batch_returns_list(self):
        bp = [-0.01, -0.005, -0.001, 0.0, 0.001, 0.003, 0.005, 0.008, 0.012]
        cands = [
            _make_candidate(raw=0.001),
            _make_candidate(raw=0.005),
        ]
        cal = EdgeCalibrator(
            table={("global",): _make_bucket(("global",), n=1000, mean=0.002)},
            decile_breakpoints=bp,
            config=CalibrationConfig(insufficient_action="global"),
        )
        result = cal.calibrate_batch(cands)
        assert len(result) == 2
        assert all(c.calibration_key == ("global",) for c in result)


# ──────────────────────────────────────────────────────────────
# from_file integration
# ──────────────────────────────────────────────────────────────
class TestFromFile:
    def test_round_trip_via_file(self, tmp_path):
        table = {
            ("global",): _make_bucket(("global",), n=10000, mean=0.002),
        }
        metadata = {
            "decile_breakpoints": [-0.01, -0.005, -0.001, 0.0, 0.001,
                                    0.003, 0.005, 0.008, 0.012],
            "version": "test",
        }
        path = tmp_path / "cal.json"
        save_calibration_table(table, path, metadata=metadata)

        cal = EdgeCalibrator.from_file(
            path,
            config=CalibrationConfig(insufficient_action="global"),
        )
        c = _make_candidate(raw=0.004)
        result = cal.calibrate(c)
        assert result.expected_return_5d == pytest.approx(0.002)

    def test_missing_decile_breakpoints_raises(self, tmp_path):
        table = {("global",): _make_bucket(("global",), n=100)}
        path = tmp_path / "cal.json"
        save_calibration_table(table, path, metadata={"version": "test"})

        with pytest.raises(ValueError, match="decile_breakpoints"):
            EdgeCalibrator.from_file(path)
