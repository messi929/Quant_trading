"""V3.3 rollback tests (P7).

Verifies:
  - compute_window_pnl_pct (date filter, missing fields, mixed formats)
  - evaluate_rollback (no degradation / triggered / no eligible activations)
  - apply_rollback_to_yaml (success, refused for NEVER_ROLLBACK, missing key, no-op)
  - load_activations / append_activation round-trip
  - append_rollback_log
  - render_rollback_alert smoke
  - End-to-end CLI (dry-run + actual)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from v3.research.rollback import (
    NEVER_ROLLBACK,
    FeatureActivation,
    RollbackDecision,
    append_activation,
    append_rollback_log,
    apply_rollback_to_yaml,
    compute_window_pnl_pct,
    evaluate_rollback,
    load_activations,
    render_rollback_alert,
)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
class TestNeverRollback:
    def test_diagnostics_protected(self):
        assert "no_trade_logger" in NEVER_ROLLBACK
        assert "tc_monitor" in NEVER_ROLLBACK
        assert "execution_quality" in NEVER_ROLLBACK
        assert len(NEVER_ROLLBACK) == 3

    def test_decision_features_not_protected(self):
        for f in ["edge_calibrator", "edge_engine", "pyramid", "rotation"]:
            assert f not in NEVER_ROLLBACK


# ──────────────────────────────────────────────────────────────
# compute_window_pnl_pct
# ──────────────────────────────────────────────────────────────
class TestWindowPnL:
    def _trade(self, exit_date: str, pnl: float) -> dict:
        return {"exit_date": exit_date, "net_pnl_krw": pnl}

    def test_empty_history(self):
        result = compute_window_pnl_pct([], pd.Timestamp("2026-05-09"), 7)
        assert result == 0.0

    def test_window_filter(self):
        today = pd.Timestamp("2026-05-09")
        trades = [
            self._trade("2026-05-08", 100_000),   # within 7d
            self._trade("2026-05-05", 50_000),    # within 7d
            self._trade("2026-04-01", 1_000_000), # outside 7d
        ]
        # Sum within window = 150,000 / 100M = 0.0015
        result = compute_window_pnl_pct(trades, today, 7, 100_000_000)
        assert result == pytest.approx(0.0015, rel=1e-3)

    def test_negative_pnl(self):
        today = pd.Timestamp("2026-05-09")
        trades = [self._trade("2026-05-08", -500_000)]
        result = compute_window_pnl_pct(trades, today, 7, 100_000_000)
        assert result == pytest.approx(-0.005)

    def test_alternative_pnl_field_names(self):
        today = pd.Timestamp("2026-05-09")
        trades = [
            {"exit_date": "2026-05-08", "realized_pnl_krw": 100_000},
            {"exit_date": "2026-05-08", "pnl_krw": 50_000},
            {"exit_date": "2026-05-08", "net_pnl_krw": 25_000},
        ]
        result = compute_window_pnl_pct(trades, today, 7, 100_000_000)
        assert result == pytest.approx(0.00175, rel=1e-3)

    def test_malformed_trades_skipped(self):
        today = pd.Timestamp("2026-05-09")
        trades = [
            {"exit_date": "not_a_date", "net_pnl_krw": 999_999},
            {"net_pnl_krw": 100_000},  # no date
            {"exit_date": "2026-05-08", "net_pnl_krw": "bad"},
            {"exit_date": "2026-05-08", "net_pnl_krw": 100_000},
        ]
        # Only last trade counted
        result = compute_window_pnl_pct(trades, today, 7, 100_000_000)
        assert result == pytest.approx(0.001)

    def test_zero_base_capital(self):
        result = compute_window_pnl_pct([], pd.Timestamp("2026-05-09"), 7, 0.0)
        assert result == 0.0


# ──────────────────────────────────────────────────────────────
# evaluate_rollback
# ──────────────────────────────────────────────────────────────
class TestEvaluateRollback:
    def _activation(self, feature: str, days_ago: int = 1) -> FeatureActivation:
        return FeatureActivation(
            feature=feature,
            activated_at=pd.Timestamp("2026-05-09") - pd.Timedelta(days=days_ago),
        )

    def test_no_degradation_no_trigger(self):
        d = evaluate_rollback(
            pnl_window_pct=0.005,  # +0.5%, above -2% threshold
            activations=[self._activation("edge_calibrator")],
        )
        assert not d.triggered
        assert d.feature_to_rollback is None

    def test_degradation_triggers_most_recent(self):
        d = evaluate_rollback(
            pnl_window_pct=-0.025,  # -2.5%
            activations=[
                self._activation("edge_calibrator", days_ago=10),
                self._activation("pyramid", days_ago=2),  # most recent
                self._activation("rotation", days_ago=5),
            ],
        )
        assert d.triggered
        assert d.feature_to_rollback == "pyramid"

    def test_diagnostics_only_no_eligible(self):
        d = evaluate_rollback(
            pnl_window_pct=-0.025,
            activations=[
                self._activation("no_trade_logger"),
                self._activation("tc_monitor"),
            ],
        )
        # Cannot rollback diagnostics
        assert not d.triggered
        assert "no rollback-eligible" in d.reason.lower()

    def test_no_activations_no_trigger(self):
        d = evaluate_rollback(
            pnl_window_pct=-0.025, activations=[],
        )
        assert not d.triggered

    def test_threshold_boundary(self):
        # exactly at threshold → not triggered (>=)
        d = evaluate_rollback(
            pnl_window_pct=-0.02,
            activations=[self._activation("pyramid")],
        )
        assert not d.triggered

    def test_custom_threshold(self):
        # -1% threshold, -1.5% pnl → triggered
        d = evaluate_rollback(
            pnl_window_pct=-0.015,
            activations=[self._activation("pyramid")],
            threshold_pct=-0.01,
        )
        assert d.triggered


# ──────────────────────────────────────────────────────────────
# apply_rollback_to_yaml
# ──────────────────────────────────────────────────────────────
class TestApplyRollback:
    def _write_config(self, tmp_path, feature_state: dict) -> Path:
        cfg_path = tmp_path / "v3_config.yaml"
        cfg = {
            "trading": {"max_positions": 3},
            "features": feature_state,
        }
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f)
        return cfg_path

    def test_success_toggles_to_false(self, tmp_path):
        cfg_path = self._write_config(tmp_path, {"pyramid": True})
        ok = apply_rollback_to_yaml(cfg_path, "pyramid")
        assert ok
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["pyramid"] is False

    def test_preserves_other_features(self, tmp_path):
        cfg_path = self._write_config(tmp_path, {
            "pyramid": True, "rotation": True, "edge_calibrator": False,
        })
        apply_rollback_to_yaml(cfg_path, "pyramid")
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["rotation"] is True
        assert cfg["features"]["edge_calibrator"] is False

    def test_refuses_never_rollback(self, tmp_path):
        cfg_path = self._write_config(tmp_path, {"no_trade_logger": True})
        ok = apply_rollback_to_yaml(cfg_path, "no_trade_logger")
        assert not ok
        # State unchanged
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["no_trade_logger"] is True

    def test_missing_feature_key(self, tmp_path):
        cfg_path = self._write_config(tmp_path, {"pyramid": True})
        ok = apply_rollback_to_yaml(cfg_path, "nonexistent_feature")
        assert not ok

    def test_already_false_no_op(self, tmp_path):
        cfg_path = self._write_config(tmp_path, {"pyramid": False})
        ok = apply_rollback_to_yaml(cfg_path, "pyramid")
        assert not ok

    def test_missing_yaml_file(self, tmp_path):
        ok = apply_rollback_to_yaml(tmp_path / "nonexistent.yaml", "pyramid")
        assert not ok


# ──────────────────────────────────────────────────────────────
# Activation history I/O
# ──────────────────────────────────────────────────────────────
class TestActivationHistory:
    def test_load_empty_when_missing(self, tmp_path):
        result = load_activations(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_round_trip(self, tmp_path):
        path = tmp_path / "activations.jsonl"
        ts1 = pd.Timestamp("2026-05-01")
        ts2 = pd.Timestamp("2026-05-08")
        append_activation("edge_calibrator", ts1, path)
        append_activation("pyramid", ts2, path, activated_by="manual")
        loaded = load_activations(path)
        assert len(loaded) == 2
        assert loaded[0].feature == "edge_calibrator"
        assert loaded[1].activated_at == ts2

    def test_load_skips_malformed(self, tmp_path):
        path = tmp_path / "activations.jsonl"
        path.write_text(
            '{"feature": "pyramid", "activated_at": "2026-05-01"}\n'
            'not_json\n'
            '{"missing": "fields"}\n'
            '{"feature": "rotation", "activated_at": "2026-05-08"}\n'
        )
        loaded = load_activations(path)
        assert len(loaded) == 2
        assert loaded[0].feature == "pyramid"
        assert loaded[1].feature == "rotation"


# ──────────────────────────────────────────────────────────────
# Rollback log + alert
# ──────────────────────────────────────────────────────────────
class TestRollbackLog:
    def test_appends(self, tmp_path):
        path = tmp_path / "rb.jsonl"
        d = RollbackDecision(
            triggered=True, reason="test",
            feature_to_rollback="pyramid",
            pnl_window_pct=-0.025, window_days=7,
        )
        append_rollback_log(d, path)
        append_rollback_log(d, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_render_triggered_alert(self):
        d = RollbackDecision(
            triggered=True, reason="pnl drop",
            feature_to_rollback="pyramid",
            pnl_window_pct=-0.025, window_days=7,
        )
        md = render_rollback_alert(d)
        assert "TRIGGERED" in md
        assert "pyramid" in md

    def test_render_no_rollback_alert(self):
        d = RollbackDecision(
            triggered=False, reason="ok",
            feature_to_rollback=None,
            pnl_window_pct=0.005, window_days=7,
        )
        md = render_rollback_alert(d)
        assert "No rollback" in md


# ──────────────────────────────────────────────────────────────
# End-to-end CLI
# ──────────────────────────────────────────────────────────────
class TestCLI:
    def test_no_paper_account_returns_2(self, tmp_path):
        from v3.scripts.run_rollback_check import main
        rc = main([
            "--paper-account", str(tmp_path / "missing.json"),
            "--config-path", str(tmp_path / "cfg.yaml"),
            "--activation-history", str(tmp_path / "act.jsonl"),
            "--log-path", str(tmp_path / "log.jsonl"),
            "--alert-dir", str(tmp_path / "alerts"),
        ])
        assert rc == 2

    def test_no_rollback_returns_0(self, tmp_path):
        from v3.scripts.run_rollback_check import main
        # Setup paper_account with positive PnL
        account_path = tmp_path / "paper_account.json"
        account_path.write_text(json.dumps({
            "cash_krw": 100_000_000,
            "positions": [],
            "trade_history": [
                {"exit_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                 "net_pnl_krw": 1_000_000},
            ],
        }))
        cfg_path = tmp_path / "cfg.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump({"features": {"pyramid": True}}, f)
        # Empty activations
        rc = main([
            "--paper-account", str(account_path),
            "--config-path", str(cfg_path),
            "--activation-history", str(tmp_path / "act.jsonl"),
            "--log-path", str(tmp_path / "log.jsonl"),
            "--alert-dir", str(tmp_path / "alerts"),
        ])
        assert rc == 0
        # Config unchanged
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["pyramid"] is True

    def test_dry_run_does_not_modify(self, tmp_path):
        from v3.scripts.run_rollback_check import main
        account_path = tmp_path / "paper_account.json"
        account_path.write_text(json.dumps({
            "cash_krw": 95_000_000,
            "positions": [],
            "trade_history": [
                {"exit_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                 "net_pnl_krw": -3_000_000},
            ],
        }))
        cfg_path = tmp_path / "cfg.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump({"features": {"pyramid": True}}, f)
        # Activation history with pyramid
        act_path = tmp_path / "act.jsonl"
        act_path.write_text(json.dumps({
            "feature": "pyramid",
            "activated_at": pd.Timestamp.now().isoformat(),
        }) + "\n")
        rc = main([
            "--paper-account", str(account_path),
            "--config-path", str(cfg_path),
            "--activation-history", str(act_path),
            "--log-path", str(tmp_path / "log.jsonl"),
            "--alert-dir", str(tmp_path / "alerts"),
            "--dry-run",
        ])
        assert rc == 1  # would have triggered
        # Config NOT modified
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["pyramid"] is True
