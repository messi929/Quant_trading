"""V3.3 execution quality monitor (PR-1.3).

Read-only DiagnosticSink. Tracks fill quality (slippage, delay, partial fill)
per ticker over time.

Limitation (paper trading):
  PaperBroker uses yfinance simulated prices. slippage_bps = decision_price vs
  simulated_fill_price, which does NOT reflect real market microstructure.
  This monitor's value emerges in production (KIS REST live execution).

Wiring (later PRs):
  - PaperBroker.execute() → record_fill() (post-execution)
  - BookOptimizer pre-execution check → is_execution_allowed() (pre-execution)
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Literal

import pandas as pd
from loguru import logger as _logger

from v3.strategy._base import DiagnosticSink


# ──────────────────────────────────────────────────────────────
# FillRecord — single execution event
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FillRecord:
    """Single execution event (buy or sell).

    decision_price: signal generation 시점 가격
    arrival_price: 주문 도착 시점 가격 (decision_price와 다를 수 있음)
    intended_order_price: 의도한 limit/market 가격
    actual_fill_price: 실제 체결가
    slippage_bps: (actual - intended) / intended × 10000, side 부호 포함
    """
    date: pd.Timestamp
    ticker: str
    side: Literal["BUY", "SELL"]
    decision_price: float
    arrival_price: float
    intended_order_price: float
    actual_fill_price: float
    fill_time: pd.Timestamp
    slippage_bps: float
    spread_proxy: float = 0.0
    execution_delay_seconds: float = 0.0
    partial_fill_ratio: float = 1.0


# ──────────────────────────────────────────────────────────────
# ExecutionQualityMonitor
# ──────────────────────────────────────────────────────────────
class ExecutionQualityMonitor:
    """Track fill quality over time per ticker.

    Implements DiagnosticSink protocol. Read-only.

    Stores recent fills in memory (deque per ticker, max recent_window) and
    persists to JSONL via flush().

    Use:
        mon = ExecutionQualityMonitor()
        mon.record_fill(FillRecord(...))
        avg_slip = mon.get_recent_slippage_bps("AAPL", lookback_days=30)
        ok, slip = mon.is_execution_allowed(net_edge=0.005, ticker="AAPL")
    """

    name = "execution_quality"

    def __init__(
        self,
        log_path: str | Path = "v3/saved_models/execution_quality.jsonl",
        recent_window: int = 100,
        enabled: bool = True,
    ):
        self.log_path = Path(log_path)
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.recent_window = recent_window
        self._enabled = enabled
        self._buffer: list[FillRecord] = []
        # Per-ticker recent fills (in-memory cache for fast slippage lookup)
        self._recent: dict[str, Deque[FillRecord]] = defaultdict(
            lambda: deque(maxlen=self.recent_window)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Recording ─────────────────────────────────────────────
    def record_fill(self, fill: FillRecord) -> None:
        """Record a single fill. Adds to buffer + per-ticker recent cache."""
        if not self._enabled:
            return
        self._buffer.append(fill)
        self._recent[fill.ticker].append(fill)

    def record(self, event: dict) -> None:
        """DiagnosticSink interface: dict → FillRecord → record_fill()."""
        if not self._enabled:
            return
        try:
            fill = FillRecord(
                date=pd.Timestamp(event["date"]),
                ticker=str(event["ticker"]),
                side=event["side"],
                decision_price=float(event["decision_price"]),
                arrival_price=float(event["arrival_price"]),
                intended_order_price=float(event["intended_order_price"]),
                actual_fill_price=float(event["actual_fill_price"]),
                fill_time=pd.Timestamp(event["fill_time"]),
                slippage_bps=float(event["slippage_bps"]),
                spread_proxy=float(event.get("spread_proxy", 0.0)),
                execution_delay_seconds=float(
                    event.get("execution_delay_seconds", 0.0)
                ),
                partial_fill_ratio=float(event.get("partial_fill_ratio", 1.0)),
            )
            self.record_fill(fill)
        except (KeyError, ValueError, TypeError) as exc:
            _logger.warning(f"ExecutionQualityMonitor.record: {exc}")

    def flush(self) -> None:
        """Append buffered fills to JSONL. Clears buffer (recent cache 유지)."""
        if not self._enabled or not self._buffer:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                for fill in self._buffer:
                    rec = {
                        "date": fill.date.isoformat(),
                        "ticker": fill.ticker,
                        "side": fill.side,
                        "decision_price": fill.decision_price,
                        "arrival_price": fill.arrival_price,
                        "intended_order_price": fill.intended_order_price,
                        "actual_fill_price": fill.actual_fill_price,
                        "fill_time": fill.fill_time.isoformat(),
                        "slippage_bps": fill.slippage_bps,
                        "spread_proxy": fill.spread_proxy,
                        "execution_delay_seconds": fill.execution_delay_seconds,
                        "partial_fill_ratio": fill.partial_fill_ratio,
                    }
                    f.write(json.dumps(rec, default=float) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            _logger.warning(f"ExecutionQualityMonitor.flush: {exc}")
            return
        self._buffer.clear()

    # ── Query ─────────────────────────────────────────────────
    def get_recent_slippage_bps(
        self,
        ticker: str,
        lookback_days: int = 30,
    ) -> float:
        """Average slippage (bps) over recent fills for a ticker.

        Returns 0.0 if no history (paper sandbox 한계 — 실거래 전엔 의미 작음).
        """
        fills = self._recent.get(ticker)
        if not fills:
            return 0.0
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent = [f.slippage_bps for f in fills if f.date >= cutoff]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def is_execution_allowed(
        self,
        net_edge: float,
        ticker: str,
        size_multiplier: float = 1.0,
    ) -> tuple[bool, float]:
        """Check if expected slippage stays within net_edge budget.

        Args:
            net_edge: 5d net edge in return units (e.g., 0.005 = 0.5%)
            ticker: lookup recent slippage
            size_multiplier: scale slippage by order size

        Returns:
            (allowed, expected_slippage_in_return_units)
            allowed = expected_slippage < net_edge
        """
        slip_bps = self.get_recent_slippage_bps(ticker)
        expected_slippage = (slip_bps / 10000.0) * size_multiplier
        return expected_slippage < net_edge, expected_slippage

    # ── Static helper ─────────────────────────────────────────
    @staticmethod
    def compute_slippage_bps(
        decision_price: float,
        actual_fill_price: float,
        side: Literal["BUY", "SELL"],
    ) -> float:
        """Compute slippage in basis points.

        BUY: positive bps when actual > decision (paid more)
        SELL: positive bps when actual < decision (received less)

        Returns 0.0 if decision_price <= 0.
        """
        if decision_price <= 0:
            return 0.0
        if side == "BUY":
            return (actual_fill_price - decision_price) / decision_price * 10000.0
        # SELL
        return (decision_price - actual_fill_price) / decision_price * 10000.0

    def buffer_size(self) -> int:
        return len(self._buffer)
