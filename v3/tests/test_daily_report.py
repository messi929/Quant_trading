"""V3.3 daily report tests (P6).

Verifies:
  - JSONL reader (empty / malformed handling)
  - compute_no_trade_summary (counts by reason / stage / ticker)
  - compute_tc_summary (today + 7d/30d trailing)
  - compute_execution_summary (avg/max slippage, per-ticker)
  - render_daily_markdown smoke
  - End-to-end CLI on synthetic logs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v3.scripts.run_daily_report import (
    build_daily_report,
    compute_execution_summary,
    compute_no_trade_summary,
    compute_tc_summary,
    main,
    render_daily_markdown,
    save_daily_report,
    _read_jsonl,
)


# ──────────────────────────────────────────────────────────────
# Synthetic logs
# ──────────────────────────────────────────────────────────────
def _write_no_trade_logs(log_dir: Path, date_str: str) -> None:
    nt_dir = log_dir / "no_trade_logs"
    nt_dir.mkdir(parents=True, exist_ok=True)
    path = nt_dir / f"no_trade_{date_str}.jsonl"
    rows = [
        {"date": date_str, "ticker": "AAPL", "stage": "edge",
         "reject_reason": "net_edge_low", "raw_opportunity": 0.001},
        {"date": date_str, "ticker": "MSFT", "stage": "edge",
         "reject_reason": "net_edge_low", "raw_opportunity": 0.0008},
        {"date": date_str, "ticker": "GOOG", "stage": "operational",
         "reject_reason": "monthly_cap", "raw_opportunity": 0.005},
        {"date": date_str, "ticker": "AAPL", "stage": "edge",
         "reject_reason": "net_edge_low", "raw_opportunity": 0.0009},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_tc_history(log_dir: Path, today: str) -> None:
    path = log_dir / "tc_history.jsonl"
    today_ts = pd.Timestamp(today)
    rows = []
    # 30 days of history
    for i in range(30):
        d = (today_ts - pd.Timedelta(days=29 - i)).strftime("%Y-%m-%d")
        rows.append({
            "date": d,
            "tc": 0.30 + 0.01 * i,  # gradual improvement
            "top_decile_capture": 0.40,
            "n_candidates": 10,
            "n_deployed": 3,
            "deployed_weight": 0.6,
        })
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_execution_quality(log_dir: Path, date_str: str) -> None:
    path = log_dir / "execution_quality.jsonl"
    rows = [
        {"date": date_str, "ticker": "AAPL", "side": "BUY",
         "decision_price": 200.0, "actual_fill_price": 200.5,
         "slippage_bps": 25.0,
         "arrival_price": 200.0, "intended_order_price": 200.0,
         "fill_time": date_str, "spread_proxy": 0.001,
         "execution_delay_seconds": 5.0, "partial_fill_ratio": 1.0},
        {"date": date_str, "ticker": "MSFT", "side": "BUY",
         "decision_price": 300.0, "actual_fill_price": 300.2,
         "slippage_bps": 6.7,
         "arrival_price": 300.0, "intended_order_price": 300.0,
         "fill_time": date_str, "spread_proxy": 0.001,
         "execution_delay_seconds": 3.0, "partial_fill_ratio": 1.0},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ──────────────────────────────────────────────────────────────
# JSONL reader
# ──────────────────────────────────────────────────────────────
class TestJSONLReader:
    def test_missing_file_returns_empty(self, tmp_path):
        rows = _read_jsonl(tmp_path / "missing.jsonl")
        assert rows == []

    def test_reads_valid_lines(self, tmp_path):
        path = tmp_path / "valid.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        rows = _read_jsonl(path)
        assert len(rows) == 2

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text('{"a": 1}\nnot_json\n{"b": 2}\n')
        rows = _read_jsonl(path)
        assert len(rows) == 2

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        rows = _read_jsonl(path)
        assert len(rows) == 2


# ──────────────────────────────────────────────────────────────
# No-trade summary
# ──────────────────────────────────────────────────────────────
class TestNoTradeSummary:
    def test_empty_when_no_logs(self, tmp_path):
        s = compute_no_trade_summary(tmp_path, "2026-05-09")
        assert s.total_rejects == 0

    def test_aggregates_by_reason(self, tmp_path):
        _write_no_trade_logs(tmp_path, "2026-05-09")
        s = compute_no_trade_summary(tmp_path, "2026-05-09")
        assert s.total_rejects == 4
        assert s.by_reason["net_edge_low"] == 3
        assert s.by_reason["monthly_cap"] == 1

    def test_top_tickers(self, tmp_path):
        _write_no_trade_logs(tmp_path, "2026-05-09")
        s = compute_no_trade_summary(tmp_path, "2026-05-09")
        # AAPL appears twice
        top = dict(s.top_rejected_tickers)
        assert top.get("AAPL") == 2


# ──────────────────────────────────────────────────────────────
# TC summary
# ──────────────────────────────────────────────────────────────
class TestTCSummary:
    def test_empty_when_no_history(self, tmp_path):
        s = compute_tc_summary(tmp_path, "2026-05-09")
        assert s.today_tc is None

    def test_today_and_trailing_means(self, tmp_path):
        today = "2026-05-09"
        _write_tc_history(tmp_path, today)
        s = compute_tc_summary(tmp_path, today)
        # today TC = 0.30 + 0.01 * 29 = 0.59
        assert s.today_tc == pytest.approx(0.59, abs=0.01)
        assert s.trailing_7d_mean_tc is not None
        assert s.trailing_30d_mean_tc is not None
        # 7d mean should be higher than 30d mean (improving series)
        assert s.trailing_7d_mean_tc > s.trailing_30d_mean_tc


# ──────────────────────────────────────────────────────────────
# Execution summary
# ──────────────────────────────────────────────────────────────
class TestExecutionSummary:
    def test_empty_when_no_fills(self, tmp_path):
        s = compute_execution_summary(tmp_path, "2026-05-09")
        assert s.n_fills == 0

    def test_aggregates_slippage(self, tmp_path):
        _write_execution_quality(tmp_path, "2026-05-09")
        s = compute_execution_summary(tmp_path, "2026-05-09")
        assert s.n_fills == 2
        # avg = (25 + 6.7) / 2 = 15.85
        assert s.avg_slippage_bps == pytest.approx(15.85, abs=0.01)
        assert s.max_slippage_bps == pytest.approx(25.0)

    def test_per_ticker(self, tmp_path):
        _write_execution_quality(tmp_path, "2026-05-09")
        s = compute_execution_summary(tmp_path, "2026-05-09")
        assert s.by_ticker_avg_slippage["AAPL"] == pytest.approx(25.0)
        assert s.by_ticker_avg_slippage["MSFT"] == pytest.approx(6.7)


# ──────────────────────────────────────────────────────────────
# Build + render
# ──────────────────────────────────────────────────────────────
class TestBuildAndRender:
    def test_build_with_all_inputs(self, tmp_path):
        date = "2026-05-09"
        _write_no_trade_logs(tmp_path, date)
        _write_tc_history(tmp_path, date)
        _write_execution_quality(tmp_path, date)
        report = build_daily_report(tmp_path, date)
        assert report.date == date
        assert report.no_trade.total_rejects == 4
        assert report.tc.today_tc is not None
        assert report.execution.n_fills == 2

    def test_build_with_no_inputs_has_notes(self, tmp_path):
        report = build_daily_report(tmp_path, "2026-05-09")
        assert len(report.notes) >= 1

    def test_render_includes_sections(self, tmp_path):
        date = "2026-05-09"
        _write_no_trade_logs(tmp_path, date)
        _write_tc_history(tmp_path, date)
        _write_execution_quality(tmp_path, date)
        report = build_daily_report(tmp_path, date)
        md = render_daily_markdown(report)
        assert "Daily Report" in md
        assert "No-Trade Reasons" in md
        assert "Transfer Coefficient" in md
        assert "Execution Quality" in md

    def test_save_writes_file(self, tmp_path):
        report = build_daily_report(tmp_path, "2026-05-09")
        out = save_daily_report(report, tmp_path / "out")
        assert out.exists()
        assert "2026-05-09.md" in out.name


# ──────────────────────────────────────────────────────────────
# CLI smoke
# ──────────────────────────────────────────────────────────────
class TestCLISmoke:
    def test_main_runs(self, tmp_path):
        date = "2026-05-09"
        _write_no_trade_logs(tmp_path / "logs", date)
        rc = main([
            "--date", date,
            "--log-dir", str(tmp_path / "logs"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert rc == 0
        assert (tmp_path / "out" / f"{date}.md").exists()
