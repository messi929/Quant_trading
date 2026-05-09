"""V3.3 diagnostics tests (PR-1.1 — NoTradeReasonLogger).

Verifies:
  - DiagnosticSink protocol satisfied
  - record() / flush() basic
  - daily JSONL grouping
  - summarize() reads disk correctly
  - disabled logger no-op
  - invalid event caught (no exception)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v3.strategy._base import DiagnosticSink
from v3.strategy.diagnostics import (
    NoTradeReasonLogger,
    RejectReason,
    TransferCoefficientMonitor,
    _rank,
    _spearman,
)
from v3.strategy.types import TCSnapshot


def _sample_event(
    ticker: str = "AAPL",
    date: str = "2026-05-09",
    stage: str = "operational",
    reason: RejectReason = RejectReason.MONTHLY_CAP,
    raw_opp: float = 0.005,
    regime: str = "neutral",
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "ticker": ticker,
        "stage": stage,
        "reject_reason": reason.value,
        "raw_opportunity": raw_opp,
        "regime": regime,
    }


# ──────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────
class TestProtocolConformance:
    def test_implements_diagnostic_sink(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        assert isinstance(log, DiagnosticSink)

    def test_name_attribute(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        assert log.name == "no_trade_logger"


# ──────────────────────────────────────────────────────────────
# Basic record + flush
# ──────────────────────────────────────────────────────────────
class TestRecordAndFlush:
    def test_record_buffers(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record(_sample_event())
        assert log.buffer_size() == 1

    def test_flush_creates_daily_jsonl(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            log.record(_sample_event(ticker=ticker, stage="edge",
                                     reason=RejectReason.NET_EDGE_TOO_LOW))
        log.flush()
        path = tmp_path / "no_trade_2026-05-09.jsonl"
        assert path.exists()
        with path.open() as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert log.buffer_size() == 0

    def test_flush_groups_by_date(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record(_sample_event(date="2026-05-09", ticker="AAPL"))
        log.record(_sample_event(date="2026-05-10", ticker="MSFT"))
        log.flush()
        assert (tmp_path / "no_trade_2026-05-09.jsonl").exists()
        assert (tmp_path / "no_trade_2026-05-10.jsonl").exists()

    def test_flush_appends_existing_file(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record(_sample_event(ticker="AAPL"))
        log.flush()
        log.record(_sample_event(ticker="MSFT"))
        log.flush()
        path = tmp_path / "no_trade_2026-05-09.jsonl"
        with path.open() as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_jsonl_format_valid(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record(_sample_event(ticker="AAPL"))
        log.flush()
        path = tmp_path / "no_trade_2026-05-09.jsonl"
        with path.open() as f:
            rec = json.loads(f.readline())
        assert rec["ticker"] == "AAPL"
        assert rec["reject_reason"] == "monthly_cap"
        assert rec["regime"] == "neutral"


# ──────────────────────────────────────────────────────────────
# Disabled / invalid event handling
# ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_disabled_logger_no_op(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path, enabled=False)
        log.record(_sample_event())
        assert log.buffer_size() == 0
        log.flush()
        # No file created
        assert not list(tmp_path.iterdir())

    def test_invalid_event_caught(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record({"ticker": "AAPL"})  # missing required fields
        # Should not raise, buffer stays empty
        assert log.buffer_size() == 0

    def test_empty_flush_safe(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.flush()  # No events buffered
        assert not list(tmp_path.iterdir())


# ──────────────────────────────────────────────────────────────
# summarize()
# ──────────────────────────────────────────────────────────────
class TestSummarize:
    def test_summarize_counts_correctly(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        for _ in range(5):
            log.record(_sample_event(stage="edge",
                                     reason=RejectReason.NET_EDGE_TOO_LOW))
        for _ in range(2):
            log.record(_sample_event(reason=RejectReason.MONTHLY_CAP))
        log.flush()

        summary = log.summarize("2026-05-09")
        assert summary["net_edge_low"] == 5
        assert summary["monthly_cap"] == 2

    def test_summarize_missing_date_returns_empty(self, tmp_path):
        log = NoTradeReasonLogger(log_dir=tmp_path)
        result = log.summarize("2026-12-31")
        assert result == {}

    def test_summarize_excludes_unflushed(self, tmp_path):
        """Summarize reads disk, not buffer."""
        log = NoTradeReasonLogger(log_dir=tmp_path)
        log.record(_sample_event())
        # Don't flush
        result = log.summarize("2026-05-09")
        assert result == {}


# ──────────────────────────────────────────────────────────────
# Taxonomy
# ──────────────────────────────────────────────────────────────
class TestRejectReasonTaxonomy:
    def test_15_reasons_defined(self):
        assert len(RejectReason) == 15

    def test_string_values(self):
        assert RejectReason.NET_EDGE_TOO_LOW.value == "net_edge_low"
        assert RejectReason.MONTHLY_CAP.value == "monthly_cap"
        assert RejectReason.RAW_OPPORTUNITY_TOO_LOW.value == "raw_opp_low"

    def test_unique_values(self):
        values = [r.value for r in RejectReason]
        assert len(values) == len(set(values))


# ──────────────────────────────────────────────────────────────
# Spearman / rank helpers
# ──────────────────────────────────────────────────────────────
class TestRankAndSpearman:
    def test_rank_strictly_increasing(self):
        assert _rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_rank_with_ties(self):
        # [1, 2, 2, 3] → ranks [1, 2.5, 2.5, 4]
        ranks = _rank([1.0, 2.0, 2.0, 3.0])
        assert ranks == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        assert _spearman(x, y) == pytest.approx(1.0)

    def test_spearman_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        assert _spearman(x, y) == pytest.approx(-1.0)

    def test_spearman_constant_series_returns_zero(self):
        x = [1, 2, 3]
        y = [5, 5, 5]
        assert _spearman(x, y) == 0.0

    def test_spearman_short_input(self):
        assert _spearman([1.0], [2.0]) == 0.0
        assert _spearman([], []) == 0.0


# ──────────────────────────────────────────────────────────────
# TransferCoefficientMonitor
# ──────────────────────────────────────────────────────────────
class _MockCandidate:
    def __init__(self, ticker: str, net_edge_5d: float):
        self.ticker = ticker
        self.net_edge_5d = net_edge_5d


class TestTransferCoefficientMonitor:
    def test_implements_diagnostic_sink(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        assert isinstance(mon, DiagnosticSink)

    def test_perfect_alignment_tc_one(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        candidates = [
            _MockCandidate("A", 0.010),
            _MockCandidate("B", 0.005),
            _MockCandidate("C", 0.001),
        ]
        weights = {"A": 0.30, "B": 0.20, "C": 0.10}
        snap = mon.compute(candidates, weights)
        assert snap.transfer_coefficient == pytest.approx(1.0)

    def test_inverse_alignment_tc_negative(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        candidates = [
            _MockCandidate("A", 0.010),
            _MockCandidate("B", 0.005),
            _MockCandidate("C", 0.001),
        ]
        weights = {"A": 0.10, "B": 0.20, "C": 0.30}
        snap = mon.compute(candidates, weights)
        assert snap.transfer_coefficient == pytest.approx(-1.0)

    def test_top_decile_capture_full_concentration(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        candidates = [_MockCandidate(f"T{i}", 0.01 - i * 0.001) for i in range(10)]
        # Only top-edge ticker T0 deployed
        weights = {f"T{i}": (0.5 if i == 0 else 0.0) for i in range(10)}
        snap = mon.compute(candidates, weights)
        assert snap.top_decile_capture == pytest.approx(1.0)

    def test_top_decile_capture_zero_when_only_low_deployed(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        candidates = [_MockCandidate(f"T{i}", 0.01 - i * 0.001) for i in range(10)]
        # Only T9 (lowest edge) deployed
        weights = {f"T{i}": (0.5 if i == 9 else 0.0) for i in range(10)}
        snap = mon.compute(candidates, weights)
        assert snap.top_decile_capture == pytest.approx(0.0)

    def test_compute_handles_empty_candidates(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        snap = mon.compute([], {})
        assert snap.transfer_coefficient == 0.0
        assert snap.top_decile_capture == 0.0
        assert snap.n_candidates == 0

    def test_alert_level_critical(self, tmp_path):
        mon = TransferCoefficientMonitor(
            log_path=tmp_path / "tc.jsonl",
            critical_threshold=0.10,
            warning_threshold=0.30,
        )
        snap = TCSnapshot(
            date=pd.Timestamp("2026-05-09"),
            transfer_coefficient=0.05,
            top_decile_capture=0.20,
            n_candidates=10,
            n_deployed=3,
            total_deployed_weight=0.6,
        )
        assert mon.alert_level(snap) == "critical"

    def test_alert_level_warning(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        snap = TCSnapshot(
            date=pd.Timestamp("2026-05-09"),
            transfer_coefficient=0.20,
            top_decile_capture=0.30,
            n_candidates=10,
            n_deployed=3,
            total_deployed_weight=0.6,
        )
        assert mon.alert_level(snap) == "warning"

    def test_alert_level_ok(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl")
        snap = TCSnapshot(
            date=pd.Timestamp("2026-05-09"),
            transfer_coefficient=0.50,
            top_decile_capture=0.55,
            n_candidates=10,
            n_deployed=3,
            total_deployed_weight=0.6,
        )
        assert mon.alert_level(snap) == "ok"

    def test_record_and_flush(self, tmp_path):
        log_path = tmp_path / "tc.jsonl"
        mon = TransferCoefficientMonitor(log_path=log_path)
        snap = TCSnapshot(
            date=pd.Timestamp("2026-05-09"),
            transfer_coefficient=0.45,
            top_decile_capture=0.55,
            n_candidates=10,
            n_deployed=3,
            total_deployed_weight=0.6,
        )
        mon.record({"snapshot": snap})
        assert mon.buffer_size() == 1
        mon.flush()
        assert mon.buffer_size() == 0
        assert log_path.exists()
        with log_path.open() as f:
            rec = json.loads(f.readline())
        assert rec["tc"] == pytest.approx(0.45)
        assert rec["top_decile_capture"] == pytest.approx(0.55)

    def test_disabled_no_op(self, tmp_path):
        mon = TransferCoefficientMonitor(log_path=tmp_path / "tc.jsonl", enabled=False)
        snap = TCSnapshot(
            date=pd.Timestamp("2026-05-09"),
            transfer_coefficient=0.45,
            top_decile_capture=0.55,
            n_candidates=10,
            n_deployed=3,
            total_deployed_weight=0.6,
        )
        mon.record({"snapshot": snap})
        assert mon.buffer_size() == 0
        mon.flush()
        assert not (tmp_path / "tc.jsonl").exists()
