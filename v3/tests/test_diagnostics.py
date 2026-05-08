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
from v3.strategy.diagnostics import NoTradeReasonLogger, RejectReason


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
