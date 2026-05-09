"""V3.3 execution quality tests (PR-1.3).

Verifies:
  - DiagnosticSink protocol satisfied
  - FillRecord frozen
  - record_fill() updates per-ticker recent cache
  - flush() persists to JSONL
  - get_recent_slippage_bps() returns mean over window
  - is_execution_allowed() blocks when slippage > net_edge
  - compute_slippage_bps() correct for BUY/SELL with sign convention
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from v3.execution.execution_quality import (
    ExecutionQualityMonitor,
    FillRecord,
)
from v3.strategy._base import DiagnosticSink


def _make_fill(
    ticker: str = "AAPL",
    date: str = "2026-05-09",
    side: str = "BUY",
    decision_price: float = 200.0,
    actual_fill_price: float = 200.5,
    slippage_bps: float = 25.0,
) -> FillRecord:
    return FillRecord(
        date=pd.Timestamp(date),
        ticker=ticker,
        side=side,
        decision_price=decision_price,
        arrival_price=decision_price,
        intended_order_price=decision_price,
        actual_fill_price=actual_fill_price,
        fill_time=pd.Timestamp(date),
        slippage_bps=slippage_bps,
    )


# ──────────────────────────────────────────────────────────────
# FillRecord
# ──────────────────────────────────────────────────────────────
class TestFillRecord:
    def test_frozen(self):
        f = _make_fill()
        with pytest.raises(FrozenInstanceError):
            f.actual_fill_price = 999.0  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────
class TestProtocolConformance:
    def test_implements_diagnostic_sink(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        assert isinstance(mon, DiagnosticSink)

    def test_name_attribute(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        assert mon.name == "execution_quality"


# ──────────────────────────────────────────────────────────────
# Record / flush
# ──────────────────────────────────────────────────────────────
class TestRecordAndFlush:
    def test_record_fill_buffers(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        mon.record_fill(_make_fill())
        assert mon.buffer_size() == 1

    def test_flush_creates_jsonl(self, tmp_path):
        log = tmp_path / "exec.jsonl"
        mon = ExecutionQualityMonitor(log_path=log)
        mon.record_fill(_make_fill(ticker="AAPL"))
        mon.record_fill(_make_fill(ticker="MSFT"))
        mon.flush()
        assert log.exists()
        with log.open() as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert mon.buffer_size() == 0

    def test_flush_appends_existing(self, tmp_path):
        log = tmp_path / "exec.jsonl"
        mon = ExecutionQualityMonitor(log_path=log)
        mon.record_fill(_make_fill(ticker="AAPL"))
        mon.flush()
        mon.record_fill(_make_fill(ticker="MSFT"))
        mon.flush()
        with log.open() as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_record_dict_interface(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        mon.record({
            "date": pd.Timestamp("2026-05-09"),
            "ticker": "AAPL",
            "side": "BUY",
            "decision_price": 200.0,
            "arrival_price": 200.0,
            "intended_order_price": 200.0,
            "actual_fill_price": 200.5,
            "fill_time": pd.Timestamp("2026-05-09"),
            "slippage_bps": 25.0,
        })
        assert mon.buffer_size() == 1

    def test_disabled_no_op(self, tmp_path):
        mon = ExecutionQualityMonitor(
            log_path=tmp_path / "exec.jsonl", enabled=False
        )
        mon.record_fill(_make_fill())
        assert mon.buffer_size() == 0
        mon.flush()
        assert not (tmp_path / "exec.jsonl").exists()

    def test_jsonl_roundtrip(self, tmp_path):
        log = tmp_path / "exec.jsonl"
        mon = ExecutionQualityMonitor(log_path=log)
        mon.record_fill(_make_fill(ticker="AAPL", slippage_bps=15.5))
        mon.flush()
        with log.open() as f:
            rec = json.loads(f.readline())
        assert rec["ticker"] == "AAPL"
        assert rec["side"] == "BUY"
        assert rec["slippage_bps"] == pytest.approx(15.5)


# ──────────────────────────────────────────────────────────────
# Recent slippage query
# ──────────────────────────────────────────────────────────────
class TestSlippageQuery:
    def test_no_history_returns_zero(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        assert mon.get_recent_slippage_bps("UNKNOWN") == 0.0

    def test_average_over_recent_fills(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        today = pd.Timestamp.now().normalize()
        for bps in [10, 20, 30, 40]:
            mon.record_fill(FillRecord(
                date=today,
                ticker="AAPL",
                side="BUY",
                decision_price=200.0,
                arrival_price=200.0,
                intended_order_price=200.0,
                actual_fill_price=200.0 + bps * 0.02,
                fill_time=today,
                slippage_bps=bps,
            ))
        avg = mon.get_recent_slippage_bps("AAPL", lookback_days=365)
        assert avg == pytest.approx(25.0)

    def test_per_ticker_isolation(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        today = pd.Timestamp.now().normalize()
        mon.record_fill(_make_fill(
            ticker="AAPL", date=today.strftime("%Y-%m-%d"), slippage_bps=10
        ))
        mon.record_fill(_make_fill(
            ticker="MSFT", date=today.strftime("%Y-%m-%d"), slippage_bps=50
        ))
        assert mon.get_recent_slippage_bps("AAPL") == pytest.approx(10.0)
        assert mon.get_recent_slippage_bps("MSFT") == pytest.approx(50.0)

    def test_recent_window_caps_cache_size(self, tmp_path):
        mon = ExecutionQualityMonitor(
            log_path=tmp_path / "exec.jsonl", recent_window=5
        )
        today = pd.Timestamp.now().normalize()
        for i in range(20):
            mon.record_fill(FillRecord(
                date=today,
                ticker="AAPL",
                side="BUY",
                decision_price=200.0,
                arrival_price=200.0,
                intended_order_price=200.0,
                actual_fill_price=200.0,
                fill_time=today,
                slippage_bps=float(i),
            ))
        # Only last 5 in cache → mean of [15,16,17,18,19] = 17
        avg = mon.get_recent_slippage_bps("AAPL")
        assert avg == pytest.approx(17.0)


# ──────────────────────────────────────────────────────────────
# is_execution_allowed
# ──────────────────────────────────────────────────────────────
class TestIsExecutionAllowed:
    def test_allowed_when_slippage_below_net_edge(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        today = pd.Timestamp.now().normalize()
        mon.record_fill(_make_fill(
            date=today.strftime("%Y-%m-%d"), slippage_bps=20
        ))
        # net_edge 0.5% > slippage 0.20% → allowed
        ok, slip = mon.is_execution_allowed(net_edge=0.005, ticker="AAPL")
        assert ok
        assert slip == pytest.approx(0.0020)

    def test_blocked_when_slippage_exceeds_net_edge(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        today = pd.Timestamp.now().normalize()
        mon.record_fill(_make_fill(
            date=today.strftime("%Y-%m-%d"), slippage_bps=60
        ))
        # net_edge 0.5% < slippage 0.60% → blocked
        ok, _ = mon.is_execution_allowed(net_edge=0.005, ticker="AAPL")
        assert not ok

    def test_unknown_ticker_allowed_with_zero_slippage(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        ok, slip = mon.is_execution_allowed(net_edge=0.005, ticker="UNKNOWN")
        assert ok
        assert slip == 0.0

    def test_size_multiplier_increases_expected_slippage(self, tmp_path):
        mon = ExecutionQualityMonitor(log_path=tmp_path / "exec.jsonl")
        today = pd.Timestamp.now().normalize()
        mon.record_fill(_make_fill(
            date=today.strftime("%Y-%m-%d"), slippage_bps=30
        ))
        # base slippage 0.30%, with 2x multiplier → 0.60%
        ok, slip = mon.is_execution_allowed(
            net_edge=0.005, ticker="AAPL", size_multiplier=2.0
        )
        assert not ok
        assert slip == pytest.approx(0.0060)


# ──────────────────────────────────────────────────────────────
# Static slippage computation
# ──────────────────────────────────────────────────────────────
class TestComputeSlippageBps:
    def test_buy_higher_than_decision_positive(self):
        # Paid 200.50 vs expected 200.00 → adverse
        bps = ExecutionQualityMonitor.compute_slippage_bps(
            decision_price=200.00, actual_fill_price=200.50, side="BUY"
        )
        # (0.50 / 200.00) * 10000 = 25
        assert bps == pytest.approx(25.0)

    def test_sell_lower_than_decision_positive(self):
        # Got 199.50 vs expected 200.00 → adverse
        bps = ExecutionQualityMonitor.compute_slippage_bps(
            decision_price=200.00, actual_fill_price=199.50, side="SELL"
        )
        assert bps == pytest.approx(25.0)

    def test_buy_lower_than_decision_negative(self):
        # Paid 199.50 vs expected 200.00 → favorable (paid less)
        bps = ExecutionQualityMonitor.compute_slippage_bps(
            decision_price=200.00, actual_fill_price=199.50, side="BUY"
        )
        assert bps == pytest.approx(-25.0)

    def test_zero_decision_price_returns_zero(self):
        bps = ExecutionQualityMonitor.compute_slippage_bps(
            decision_price=0.0, actual_fill_price=200.0, side="BUY"
        )
        assert bps == 0.0
