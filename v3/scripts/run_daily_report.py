"""V3.3 daily diagnostic report (P6).

Aggregates V3.3 diagnostic sinks into a single daily markdown:
  - NoTradeReasonLogger: per-reason counts
  - TransferCoefficientMonitor: today vs trailing average
  - ExecutionQualityMonitor: slippage stats
  - paper_account.json: trades / open positions / realized PnL

Production cron entry (KST 16:00 after market close):
  0 16 * * * /opt/quant/run_daily_report.sh

Inputs (paths from v3/config/v3_config.yaml):
  v3/saved_models/no_trade_logs/no_trade_<YYYY-MM-DD>.jsonl
  v3/saved_models/tc_history.jsonl
  v3/saved_models/execution_quality.jsonl
  v3/saved_models/paper_account.json
  v3/saved_models/recommendation_log.jsonl

Outputs:
  research/reports/daily/<YYYY-MM-DD>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NoTradeSummary:
    total_rejects: int
    by_reason: dict[str, int]
    by_stage: dict[str, int]
    top_rejected_tickers: list[tuple[str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class TCSummary:
    today_tc: Optional[float]
    today_capture: Optional[float]
    today_n_candidates: Optional[int]
    today_n_deployed: Optional[int]
    trailing_7d_mean_tc: Optional[float]
    trailing_30d_mean_tc: Optional[float]


@dataclass(frozen=True)
class ExecutionSummary:
    n_fills: int
    avg_slippage_bps: Optional[float]
    max_slippage_bps: Optional[float]
    by_ticker_avg_slippage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DailyReport:
    date: str
    no_trade: NoTradeSummary
    tc: TCSummary
    execution: ExecutionSummary
    notes: tuple[str, ...] = field(default_factory=tuple)


# ──────────────────────────────────────────────────────────────
# JSONL reader
# ──────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _filter_by_date(rows: list[dict], date_str: str, key: str = "date") -> list[dict]:
    """Keep rows where row[key] starts with date_str (YYYY-MM-DD)."""
    out: list[dict] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if str(v).startswith(date_str):
            out.append(r)
    return out


# ──────────────────────────────────────────────────────────────
# Section computations
# ──────────────────────────────────────────────────────────────
def compute_no_trade_summary(
    log_dir: Path,
    date_str: str,
) -> NoTradeSummary:
    path = log_dir / "no_trade_logs" / f"no_trade_{date_str}.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return NoTradeSummary(total_rejects=0, by_reason={}, by_stage={})

    by_reason = Counter(r.get("reject_reason", "unknown") for r in rows)
    by_stage = Counter(r.get("stage", "unknown") for r in rows)
    by_ticker = Counter(r.get("ticker", "?") for r in rows)
    top_tickers = by_ticker.most_common(5)
    return NoTradeSummary(
        total_rejects=len(rows),
        by_reason=dict(by_reason),
        by_stage=dict(by_stage),
        top_rejected_tickers=top_tickers,
    )


def compute_tc_summary(
    log_dir: Path,
    date_str: str,
    trailing_7d_days: int = 7,
    trailing_30d_days: int = 30,
) -> TCSummary:
    path = log_dir / "tc_history.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return TCSummary(
            today_tc=None, today_capture=None,
            today_n_candidates=None, today_n_deployed=None,
            trailing_7d_mean_tc=None, trailing_30d_mean_tc=None,
        )

    today_rows = _filter_by_date(rows, date_str)
    today = today_rows[-1] if today_rows else None

    # Trailing windows
    end_date = pd.Timestamp(date_str)
    start_7d = end_date - pd.Timedelta(days=trailing_7d_days)
    start_30d = end_date - pd.Timedelta(days=trailing_30d_days)

    def _in_window(row: dict, start: pd.Timestamp) -> bool:
        try:
            ts = pd.Timestamp(row.get("date"))
            return start <= ts <= end_date
        except (ValueError, TypeError):
            return False

    rows_7d = [r for r in rows if _in_window(r, start_7d)]
    rows_30d = [r for r in rows if _in_window(r, start_30d)]
    mean_7d = (
        sum(r.get("tc", 0.0) for r in rows_7d) / len(rows_7d)
        if rows_7d else None
    )
    mean_30d = (
        sum(r.get("tc", 0.0) for r in rows_30d) / len(rows_30d)
        if rows_30d else None
    )

    return TCSummary(
        today_tc=today.get("tc") if today else None,
        today_capture=today.get("top_decile_capture") if today else None,
        today_n_candidates=today.get("n_candidates") if today else None,
        today_n_deployed=today.get("n_deployed") if today else None,
        trailing_7d_mean_tc=mean_7d,
        trailing_30d_mean_tc=mean_30d,
    )


def compute_execution_summary(
    log_dir: Path,
    date_str: str,
) -> ExecutionSummary:
    path = log_dir / "execution_quality.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return ExecutionSummary(
            n_fills=0, avg_slippage_bps=None, max_slippage_bps=None,
        )

    today_rows = _filter_by_date(rows, date_str)
    if not today_rows:
        return ExecutionSummary(
            n_fills=0, avg_slippage_bps=None, max_slippage_bps=None,
        )

    slippages = [float(r.get("slippage_bps", 0.0)) for r in today_rows]
    avg = sum(slippages) / len(slippages) if slippages else None
    mx = max(slippages, default=None)

    by_ticker: dict[str, list[float]] = {}
    for r in today_rows:
        t = r.get("ticker", "?")
        by_ticker.setdefault(t, []).append(float(r.get("slippage_bps", 0.0)))
    by_ticker_avg = {t: sum(v) / len(v) for t, v in by_ticker.items()}

    return ExecutionSummary(
        n_fills=len(today_rows),
        avg_slippage_bps=avg,
        max_slippage_bps=mx,
        by_ticker_avg_slippage=by_ticker_avg,
    )


# ──────────────────────────────────────────────────────────────
# Report assembly
# ──────────────────────────────────────────────────────────────
def build_daily_report(
    log_dir: Path,
    date_str: str,
) -> DailyReport:
    notes: list[str] = []

    no_trade = compute_no_trade_summary(log_dir, date_str)
    tc = compute_tc_summary(log_dir, date_str)
    execution = compute_execution_summary(log_dir, date_str)

    if no_trade.total_rejects == 0:
        notes.append("No no_trade events recorded — feature.no_trade_logger may be OFF.")
    if tc.today_tc is None:
        notes.append("No TC snapshot — feature.tc_monitor may be OFF.")
    if execution.n_fills == 0:
        notes.append("No fills recorded today — quiet day or execution_quality OFF.")

    return DailyReport(
        date=date_str,
        no_trade=no_trade,
        tc=tc,
        execution=execution,
        notes=tuple(notes),
    )


def render_daily_markdown(report: DailyReport) -> str:
    lines: list[str] = []
    lines.append(f"# V3.3 Daily Report — {report.date}")
    lines.append("")

    # No-trade summary
    lines.append("## No-Trade Reasons")
    lines.append("")
    if report.no_trade.total_rejects == 0:
        lines.append("_No no_trade events recorded._")
    else:
        lines.append(f"- Total rejects: {report.no_trade.total_rejects}")
        lines.append("")
        lines.append("**By reason:**")
        for reason, n in sorted(
            report.no_trade.by_reason.items(), key=lambda x: -x[1]
        ):
            pct = n / report.no_trade.total_rejects * 100.0
            lines.append(f"- `{reason}`: {n} ({pct:.0f}%)")
        lines.append("")
        lines.append("**By stage:**")
        for stage, n in sorted(
            report.no_trade.by_stage.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {stage}: {n}")
        if report.no_trade.top_rejected_tickers:
            lines.append("")
            lines.append("**Top rejected tickers:**")
            for ticker, n in report.no_trade.top_rejected_tickers:
                lines.append(f"- {ticker}: {n}")
    lines.append("")

    # TC summary
    lines.append("## Transfer Coefficient")
    lines.append("")
    if report.tc.today_tc is None:
        lines.append("_No TC snapshot for today._")
    else:
        lines.append(f"- Today TC: **{report.tc.today_tc:.3f}**")
        lines.append(f"- Today top-decile capture: {report.tc.today_capture:.3f}")
        lines.append(
            f"- Today candidates / deployed: "
            f"{report.tc.today_n_candidates} / {report.tc.today_n_deployed}"
        )
        if report.tc.trailing_7d_mean_tc is not None:
            lines.append(f"- 7-day mean TC: {report.tc.trailing_7d_mean_tc:.3f}")
        if report.tc.trailing_30d_mean_tc is not None:
            lines.append(f"- 30-day mean TC: {report.tc.trailing_30d_mean_tc:.3f}")
        if report.tc.today_tc < 0.10:
            lines.append("- ⚠ TC critical (< 0.10)")
        elif report.tc.today_tc < 0.30:
            lines.append("- ⚠ TC warning (< 0.30)")
    lines.append("")

    # Execution
    lines.append("## Execution Quality")
    lines.append("")
    if report.execution.n_fills == 0:
        lines.append("_No fills today._")
    else:
        lines.append(f"- Fills: {report.execution.n_fills}")
        lines.append(f"- Avg slippage: {report.execution.avg_slippage_bps:.2f} bps")
        lines.append(f"- Max slippage: {report.execution.max_slippage_bps:.2f} bps")
        if report.execution.by_ticker_avg_slippage:
            lines.append("")
            lines.append("**By ticker:**")
            for t, slip in sorted(
                report.execution.by_ticker_avg_slippage.items(),
                key=lambda x: -abs(x[1]),
            ):
                lines.append(f"- {t}: {slip:.2f} bps")
    lines.append("")

    # Notes
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for n in report.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


def save_daily_report(
    report: DailyReport,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.date}.md"
    path.write_text(render_daily_markdown(report), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--log-dir", type=Path,
        default=Path("v3/saved_models"),
        help="Directory containing no_trade_logs/, tc_history.jsonl, etc.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("research/reports/daily"),
        help="Where to write daily markdown",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    date_str = args.date or pd.Timestamp.now().strftime("%Y-%m-%d")
    logger.info(f"Building daily report for {date_str}")

    report = build_daily_report(args.log_dir, date_str)
    out_path = save_daily_report(report, args.output_dir)
    logger.info(f"Saved daily report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
