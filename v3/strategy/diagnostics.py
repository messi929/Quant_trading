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
from v3.strategy.types import NoTradeLog, TCSnapshot


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


# ──────────────────────────────────────────────────────────────
# TransferCoefficientMonitor (PR-1.2)
# ──────────────────────────────────────────────────────────────
def _rank(values: list[float]) -> list[float]:
    """Average ranks (1-indexed, handles ties).

    No scipy dependency. Tied values get average rank.
    """
    n = len(values)
    if n == 0:
        return []
    sorted_pairs = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. No scipy dependency.

    Returns 0.0 for degenerate cases (n<2, constant series).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry)) / n
    var_rx = sum((a - mean_rx) ** 2 for a in rx) / n
    var_ry = sum((b - mean_ry) ** 2 for b in ry) / n
    if var_rx <= 0 or var_ry <= 0:
        return 0.0
    return cov / ((var_rx ** 0.5) * (var_ry ** 0.5))


class TransferCoefficientMonitor:
    """Edge-rank vs weight-rank correlation tracking.

    Implements DiagnosticSink protocol. Read-only.

    transfer_coefficient = Spearman corr (net_edge_rank, final_weight_rank)
    top_decile_capture   = weight in top edge decile / total deployed weight

    Threshold semantics (default):
      tc < 0.10  → critical: signals not reaching portfolio
      tc < 0.30  → warning: weak signal-to-weight transfer
      tc ≥ 0.30  → ok

    Use:
        mon = TransferCoefficientMonitor()
        snap = mon.compute(candidates, final_weights)
        mon.record({"snapshot": snap})
        mon.flush()
    """

    name = "tc_monitor"

    def __init__(
        self,
        log_path: str | Path = "v3/saved_models/tc_history.jsonl",
        warning_threshold: float = 0.30,
        critical_threshold: float = 0.10,
        top_decile_capture_min: float = 0.40,
        enabled: bool = True,
    ):
        self.log_path = Path(log_path)
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.top_decile_capture_min = top_decile_capture_min
        self._enabled = enabled
        self._buffer: list[TCSnapshot] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def compute(
        self,
        candidates: list,
        final_weights: dict[str, float],
        as_of: pd.Timestamp | None = None,
    ) -> TCSnapshot:
        """Compute TC + top decile capture from candidates and weights.

        Args:
            candidates: list of objects with `.ticker` and `.net_edge_5d` attrs,
                or dicts with same keys. None values treated as 0.
            final_weights: {ticker: weight}. Weights ≤ 0 treated as not deployed.
            as_of: snapshot timestamp. Defaults to today (normalized).

        Returns:
            Immutable TCSnapshot.
        """
        as_of = as_of if as_of is not None else pd.Timestamp.now().normalize()

        edges: list[float] = []
        tickers: list[str] = []
        for c in candidates:
            ticker = getattr(c, "ticker", None)
            if ticker is None and isinstance(c, dict):
                ticker = c.get("ticker", "")
            edge = getattr(c, "net_edge_5d", None)
            if edge is None and isinstance(c, dict):
                edge = c.get("net_edge_5d")
            tickers.append(str(ticker or ""))
            edges.append(float(edge) if edge is not None else 0.0)

        weights_aligned = [float(final_weights.get(t, 0.0)) for t in tickers]

        n_candidates = len(candidates)
        deployed_weight = sum(w for w in final_weights.values() if w > 0)
        n_deployed = sum(1 for w in final_weights.values() if w > 0)

        # Spearman correlation
        tc = _spearman(edges, weights_aligned) if n_candidates > 1 else 0.0

        # Top decile capture
        if n_candidates == 0 or deployed_weight <= 0:
            capture = 0.0
        else:
            top_n = max(1, n_candidates // 10)
            sorted_idx = sorted(
                range(n_candidates), key=lambda i: edges[i], reverse=True
            )
            top_tickers = {tickers[i] for i in sorted_idx[:top_n]}
            top_weight = sum(
                w for t, w in final_weights.items()
                if t in top_tickers and w > 0
            )
            capture = top_weight / deployed_weight

        return TCSnapshot(
            date=as_of,
            transfer_coefficient=tc,
            top_decile_capture=capture,
            n_candidates=n_candidates,
            n_deployed=n_deployed,
            total_deployed_weight=deployed_weight,
        )

    def record(self, event: dict) -> None:
        """Buffer a TCSnapshot.

        Two formats accepted:
          {"snapshot": TCSnapshot(...)}  — preferred
          {"date": ..., "transfer_coefficient": ..., ...}  — manual fields
        """
        if not self._enabled:
            return
        try:
            snap = event.get("snapshot")
            if isinstance(snap, TCSnapshot):
                self._buffer.append(snap)
                return
            self._buffer.append(TCSnapshot(
                date=pd.Timestamp(event["date"]),
                transfer_coefficient=float(event["transfer_coefficient"]),
                top_decile_capture=float(event["top_decile_capture"]),
                n_candidates=int(event["n_candidates"]),
                n_deployed=int(event["n_deployed"]),
                total_deployed_weight=float(event["total_deployed_weight"]),
            ))
        except (KeyError, ValueError, TypeError) as exc:
            _logger.warning(f"TransferCoefficientMonitor.record: {exc}")

    def flush(self) -> None:
        """Append buffered snapshots to JSONL. Clears buffer."""
        if not self._enabled or not self._buffer:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                for snap in self._buffer:
                    rec = {
                        "date": snap.date.isoformat(),
                        "tc": snap.transfer_coefficient,
                        "top_decile_capture": snap.top_decile_capture,
                        "n_candidates": snap.n_candidates,
                        "n_deployed": snap.n_deployed,
                        "deployed_weight": snap.total_deployed_weight,
                    }
                    f.write(json.dumps(rec, default=float) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            _logger.warning(f"TransferCoefficientMonitor.flush: {exc}")
            return
        self._buffer.clear()

    def alert_level(self, snapshot: TCSnapshot) -> str:
        """Classify snapshot severity. Returns 'critical' | 'warning' | 'ok'."""
        if snapshot.transfer_coefficient < self.critical_threshold:
            return "critical"
        if snapshot.transfer_coefficient < self.warning_threshold:
            return "warning"
        return "ok"

    def buffer_size(self) -> int:
        return len(self._buffer)
