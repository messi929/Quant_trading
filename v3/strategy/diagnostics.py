"""V3.3 read-only diagnostic sinks (Phase 1).

Modules:
  - NoTradeReasonLogger (PR-1.1): per-candidate rejection tracking
  - TransferCoefficientMonitor (PR-1.2): edge_rank vs weight_rank correlation

Both implement DiagnosticSink protocol from v3/strategy/_base.py.
Read-only — does NOT affect decisions.

Wiring strategy:
  - Phase 1: standalone modules, callable from any pipeline
  - Phase 2 (PR-2.4 BookOptimizer): integrated as constructor injection
  - Until then, NoTradeReasonLogger is invoked manually in research scripts

Coexistence with v3/pipeline/live_pipeline.py recommendation_log:
  - recommendation_log: session-level snapshot (1 line per session, macro view)
  - NoTradeReasonLogger: candidate-level (per-rejection, micro view)
  - Two are complementary, not redundant. Both kept.
"""

from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from pathlib import Path

import pandas as pd
from loguru import logger as _logger

from v3.strategy._base import DiagnosticSink
from v3.strategy.types import NoTradeLog


# ──────────────────────────────────────────────────────────────
# Reject reason taxonomy (15 reasons covering all V3.3 stages)
# ──────────────────────────────────────────────────────────────
class RejectReason(str, Enum):
    """Stage별 reject 분류. 모든 reject는 정확히 1개 reason 가져야 한다.

    String values used in JSONL persistence and summarize() output.
    """
    # Stage: opportunity (raw signal — V3.2.1 OpportunityScorer)
    RAW_OPPORTUNITY_TOO_LOW = "raw_opp_low"

    # Stage: calibration (Phase 2 EdgeCalibrator)
    INSUFFICIENT_CALIBRATION = "calib_insufficient"

    # Stage: edge (Phase 2 EdgeEngine)
    EXPECTED_RETURN_TOO_LOW = "exp_return_low"
    NET_EDGE_TOO_LOW = "net_edge_low"
    WIN_PROB_TOO_LOW = "win_prob_low"
    EXPECTED_MAE_TOO_LARGE = "exp_mae_large"

    # Stage: tier (Phase 2 EdgeTierSystem)
    TIER_BELOW_MIN = "tier_low"

    # Stage: operational (V3.2.1 EntryFilter — already enforced)
    POSITION_LIMIT = "pos_limit"
    MONTHLY_CAP = "monthly_cap"
    LIQUIDITY_LIMIT = "liquidity"
    SECTOR_LIMIT = "sector"
    MIN_ORDER_AMOUNT = "min_order"
    REGIME_CASH_ONLY = "regime_cash"

    # Stage: execution (Phase 2/3)
    EXECUTION_COST_TOO_HIGH = "exec_cost"
    NO_FRESH_SIGNAL = "no_fresh_signal"


# ──────────────────────────────────────────────────────────────
# NoTradeReasonLogger
# ──────────────────────────────────────────────────────────────
class NoTradeReasonLogger:
    """Per-candidate rejection tracking with daily JSONL persistence.

    Implements DiagnosticSink protocol. Read-only — does not affect decisions.

    Buffered writes — call flush() at session boundary.
    Daily files: v3/saved_models/no_trade_logs/no_trade_YYYY-MM-DD.jsonl

    Usage:
        log = NoTradeReasonLogger()
        log.record({
            "date": pd.Timestamp("2026-05-09"),
            "ticker": "AAPL",
            "stage": "operational",
            "reject_reason": RejectReason.MONTHLY_CAP.value,
            "raw_opportunity": 0.005,
            "regime": "neutral",
            "details": {"current_count": 5, "limit": 5},
        })
        log.flush()
    """

    name = "no_trade_logger"

    def __init__(
        self,
        log_dir: str | Path = "v3/saved_models/no_trade_logs",
        enabled: bool = True,
    ):
        self.log_dir = Path(log_dir)
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled
        self._buffer: list[NoTradeLog] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, event: dict) -> None:
        """Buffer a single rejection event.

        Required fields: date, ticker, stage, reject_reason, raw_opportunity.
        Optional: regime, net_edge_5d, edge_tier, details.

        Invalid events are dropped with a warning (no exception raised).
        """
        if not self._enabled:
            return
        try:
            log = NoTradeLog(
                date=pd.Timestamp(event["date"]),
                ticker=str(event["ticker"]),
                stage=str(event["stage"]),
                reject_reason=str(event["reject_reason"]),
                raw_opportunity=float(event["raw_opportunity"]),
                regime=str(event.get("regime", "")),
                net_edge_5d=event.get("net_edge_5d"),
                edge_tier=event.get("edge_tier"),
                details=dict(event.get("details", {})),
            )
            self._buffer.append(log)
        except (KeyError, ValueError, TypeError) as exc:
            _logger.warning(f"NoTradeReasonLogger.record: invalid event ({exc})")

    def flush(self) -> None:
        """Persist buffered events to daily JSONL files. Clears buffer."""
        if not self._enabled or not self._buffer:
            return

        by_date: dict[str, list[NoTradeLog]] = defaultdict(list)
        for log in self._buffer:
            by_date[log.date.strftime("%Y-%m-%d")].append(log)

        for date_str, logs in by_date.items():
            path = self.log_dir / f"no_trade_{date_str}.jsonl"
            try:
                with path.open("a", encoding="utf-8") as f:
                    for log in logs:
                        rec = {
                            "date": log.date.isoformat(),
                            "ticker": log.ticker,
                            "stage": log.stage,
                            "reject_reason": log.reject_reason,
                            "raw_opportunity": log.raw_opportunity,
                            "regime": log.regime,
                            "net_edge_5d": log.net_edge_5d,
                            "edge_tier": log.edge_tier,
                            "details": log.details,
                        }
                        f.write(
                            json.dumps(rec, ensure_ascii=False, default=float) + "\n"
                        )
            except (OSError, TypeError, ValueError) as exc:
                _logger.warning(f"NoTradeReasonLogger.flush: {exc}")
                continue

        self._buffer.clear()

    def summarize(self, date: str) -> dict[str, int]:
        """Daily reject reason distribution (count per reason).

        Reads from disk — independent of in-memory buffer.
        Call after flush() to ensure latest data persisted.

        Args:
            date: ISO format date string (YYYY-MM-DD)

        Returns:
            {reject_reason: count}. Empty dict if no log file.
        """
        path = self.log_dir / f"no_trade_{date}.jsonl"
        if not path.exists():
            return {}
        counts: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    reason = rec.get("reject_reason", "unknown")
                    counts[reason] = counts.get(reason, 0) + 1
                except json.JSONDecodeError:
                    continue
        return counts

    def buffer_size(self) -> int:
        """Current in-memory buffer size (for testing)."""
        return len(self._buffer)
