"""V3.3 feature_tracker tests (F2 fix).

Verifies:
  - _load_snapshot / _save_snapshot round-trip
  - detect_newly_activated (no snapshot / no change / single toggle / multiple)
  - record_features_on_startup end-to-end (idempotent, JSONL grows)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v3.config.schema import FeatureFlagsConfig
from v3.research.rollback import load_activations
from v3.strategy.feature_tracker import (
    _load_snapshot,
    _save_snapshot,
    detect_newly_activated,
    record_features_on_startup,
)


# ──────────────────────────────────────────────────────────────
# Snapshot I/O
# ──────────────────────────────────────────────────────────────
class TestSnapshot:
    def test_load_missing_returns_empty(self, tmp_path):
        result = _load_snapshot(tmp_path / "nonexistent.json")
        assert result == {}

    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "snap.json"
        ts = pd.Timestamp("2026-05-09T16:00:00")
        features = {"no_trade_logger": True, "edge_calibrator": False}
        _save_snapshot(path, features, ts)
        loaded = _load_snapshot(path)
        assert loaded["features"] == features
        assert loaded["last_seen"] == ts.isoformat()

    def test_load_malformed_returns_empty(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{invalid json")
        result = _load_snapshot(path)
        assert result == {}


# ──────────────────────────────────────────────────────────────
# detect_newly_activated
# ──────────────────────────────────────────────────────────────
class TestDetectNewlyActivated:
    def test_no_snapshot_treats_all_active_as_new(self):
        current = FeatureFlagsConfig(no_trade_logger=True, edge_calibrator=True)
        result = detect_newly_activated(current, snapshot={})
        assert "no_trade_logger" in result
        assert "edge_calibrator" in result
        # Inactive features not included
        assert "pyramid" not in result

    def test_no_change_returns_empty(self):
        current = FeatureFlagsConfig(no_trade_logger=True)
        snapshot = {"features": {"no_trade_logger": True}}
        result = detect_newly_activated(current, snapshot)
        assert result == []

    def test_single_toggle(self):
        current = FeatureFlagsConfig(
            no_trade_logger=True, edge_calibrator=True,
        )
        snapshot = {"features": {
            "no_trade_logger": True, "edge_calibrator": False,
        }}
        result = detect_newly_activated(current, snapshot)
        assert result == ["edge_calibrator"]

    def test_multiple_toggles_sorted(self):
        current = FeatureFlagsConfig(
            no_trade_logger=True, edge_calibrator=True, edge_engine=True,
        )
        snapshot = {"features": {}}
        result = detect_newly_activated(current, snapshot)
        # Should include all 3, sorted
        assert "edge_calibrator" in result
        assert "edge_engine" in result
        assert "no_trade_logger" in result
        assert result == sorted(result)

    def test_deactivation_not_recorded(self):
        # true → false is NOT a new activation
        current = FeatureFlagsConfig(no_trade_logger=False)
        snapshot = {"features": {"no_trade_logger": True}}
        result = detect_newly_activated(current, snapshot)
        assert result == []


# ──────────────────────────────────────────────────────────────
# record_features_on_startup end-to-end
# ──────────────────────────────────────────────────────────────
class TestRecordOnStartup:
    def test_first_startup_records_active_features(self, tmp_path):
        history = tmp_path / "activations.jsonl"
        snapshot = tmp_path / "snap.json"
        current = FeatureFlagsConfig(no_trade_logger=True, edge_calibrator=True)

        n = record_features_on_startup(
            current=current,
            history_path=history,
            snapshot_path=snapshot,
            now=pd.Timestamp("2026-05-09T09:30:00"),
        )
        assert n == 2

        # History has both
        loaded = load_activations(history)
        features_recorded = {a.feature for a in loaded}
        assert features_recorded == {"no_trade_logger", "edge_calibrator"}

        # Snapshot saved
        snap = _load_snapshot(snapshot)
        assert snap["features"]["no_trade_logger"] is True

    def test_idempotent_no_change(self, tmp_path):
        history = tmp_path / "activations.jsonl"
        snapshot = tmp_path / "snap.json"
        current = FeatureFlagsConfig(no_trade_logger=True)

        n1 = record_features_on_startup(
            current=current, history_path=history, snapshot_path=snapshot,
            now=pd.Timestamp("2026-05-09T09:30:00"),
        )
        n2 = record_features_on_startup(
            current=current, history_path=history, snapshot_path=snapshot,
            now=pd.Timestamp("2026-05-09T15:30:00"),  # same day later
        )
        assert n1 == 1
        assert n2 == 0  # no change

        loaded = load_activations(history)
        assert len(loaded) == 1  # only first call recorded

    def test_subsequent_toggle_recorded(self, tmp_path):
        history = tmp_path / "activations.jsonl"
        snapshot = tmp_path / "snap.json"

        # First startup: only diagnostics ON
        record_features_on_startup(
            current=FeatureFlagsConfig(no_trade_logger=True),
            history_path=history, snapshot_path=snapshot,
            now=pd.Timestamp("2026-05-15T09:30:00"),
        )
        # Second startup: edge_calibrator added
        n = record_features_on_startup(
            current=FeatureFlagsConfig(
                no_trade_logger=True, edge_calibrator=True,
            ),
            history_path=history, snapshot_path=snapshot,
            now=pd.Timestamp("2026-05-22T09:30:00"),
        )
        assert n == 1

        loaded = load_activations(history)
        names = [a.feature for a in loaded]
        assert "no_trade_logger" in names
        assert "edge_calibrator" in names

    def test_activated_by_recorded(self, tmp_path):
        history = tmp_path / "activations.jsonl"
        snapshot = tmp_path / "snap.json"
        record_features_on_startup(
            current=FeatureFlagsConfig(pyramid=True),
            history_path=history, snapshot_path=snapshot,
            activated_by="manual_test",
            now=pd.Timestamp("2026-05-09"),
        )
        loaded = load_activations(history)
        assert loaded[0].activated_by == "manual_test"

    def test_all_off_no_recording(self, tmp_path):
        history = tmp_path / "activations.jsonl"
        snapshot = tmp_path / "snap.json"
        n = record_features_on_startup(
            current=FeatureFlagsConfig(),  # all off
            history_path=history, snapshot_path=snapshot,
        )
        assert n == 0
        # Snapshot still saved (even with all-off features)
        assert snapshot.exists()
        # No history file (or empty)
        if history.exists():
            assert history.read_text().strip() == ""
