"""V3.3 monthly calibration pipeline (P5).

End-to-end pipeline: build → calibrate → validate → archive.

Production cron entry (server, monthly):
  0 7 1 * * /opt/quant/run_calibration.sh

run_calibration.sh wraps:
  PYTHONPATH=/opt/quant /opt/quant/venv/bin/python \\
      /opt/quant/v3/scripts/run_calibration_pipeline.py \\
      --ohlcv-path data/research/ohlcv_panel.parquet \\
      --macro-path data/research/macro_pctl.parquet \\
      --vol-pred-path data/research/vol_predictions.parquet \\
      --years-back 5

Outputs:
  v3/config/edge_calibration.json                      (latest, used by runtime)
  v3/config/edge_calibration_history/<YYYY-MM>.json    (versioned archive)
  research/reports/validation_<YYYY-MM>.md             (OOS pass/fail report)

Exit codes:
  0  validation passed
  1  validation failed (OOS metrics below thresholds)
  2  data missing or pipeline error
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from v3.research.build_edge_dataset import (
    BuildConfig,
    build_edge_dataset,
    panel_sanity_summary,
)
from v3.research.calibrate_edge import calibrate_panel
from v3.research.validate_edge import (
    render_markdown_report,
    validate_oos,
)


# ──────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────
def _resolve_dates(years_back: int, oos_days: int) -> tuple[str, str, str]:
    """Compute (start, train_end, end) date strings."""
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=365 * years_back)
    train_end = end - pd.Timedelta(days=oos_days)
    return (
        start.strftime("%Y-%m-%d"),
        train_end.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def _step_build(
    args: argparse.Namespace,
    cfg: BuildConfig,
    panel_path: Path,
) -> tuple[bool, str | None]:
    """Build edge_dataset panel from production data."""
    if not all([args.ohlcv_path, args.macro_path, args.vol_pred_path]):
        return False, (
            "Missing required data paths. Production mode requires:\n"
            "  --ohlcv-path / --macro-path / --vol-pred-path"
        )

    for p in (args.ohlcv_path, args.macro_path, args.vol_pred_path):
        if not p.exists():
            return False, f"Data file not found: {p}"

    logger.info(
        f"Loading inputs:\n"
        f"  OHLCV: {args.ohlcv_path}\n"
        f"  Macro: {args.macro_path}\n"
        f"  Vol preds: {args.vol_pred_path}"
    )
    ohlcv = pd.read_parquet(args.ohlcv_path)
    macro = pd.read_parquet(args.macro_path)
    vol_pred = pd.read_parquet(args.vol_pred_path)

    panel = build_edge_dataset(
        cfg,
        ohlcv_data=ohlcv,
        macro_pctl=macro,
        vol_predictions=vol_pred,
    )
    summary = panel_sanity_summary(panel)
    logger.info(f"Panel summary: {summary}")

    if panel.empty:
        return False, "build_edge_dataset returned empty panel"

    # Check minimum sample sizes
    n_rows = summary.get("n_rows", 0)
    if n_rows < 5000:
        logger.warning(
            f"Panel size {n_rows} < 5000 — calibration may be unreliable"
        )

    return True, None


def _step_calibrate(
    panel_path: Path,
    calib_path: Path,
    train_end: str,
    min_bucket_n: int,
) -> tuple[bool, str | None]:
    """Run calibrate_panel CLI logic."""
    try:
        summary = calibrate_panel(
            panel_path=panel_path,
            output_path=calib_path,
            train_end=train_end,
            min_bucket_n=min_bucket_n,
        )
        logger.info(f"Calibration summary: {summary}")
        if summary["n_buckets"] == 0:
            return False, "Zero buckets produced — check min_bucket_n"
        return True, None
    except Exception as exc:
        return False, f"Calibration error: {exc!r}"


def _step_validate(
    panel_path: Path,
    calib_path: Path,
    train_end: str,
    report_path: Path,
) -> tuple[bool, bool, str | None]:
    """Run validate_oos and render markdown.

    Returns:
        (success, validation_passed, error_message)
        success: pipeline ran without exception
        validation_passed: OOS metrics passed thresholds
    """
    try:
        result = validate_oos(
            panel_path=panel_path,
            calibration_path=calib_path,
            train_end=train_end,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_markdown_report(result), encoding="utf-8"
        )
        logger.info(
            f"Validation: monotonicity={result.decile_monotonicity_spearman:.3f}, "
            f"top_decile_positive={result.top_decile_mean_positive}, "
            f"PASS={result.passed}"
        )
        return True, result.passed, None
    except Exception as exc:
        return False, False, f"Validation error: {exc!r}"


def _step_archive(
    calib_path: Path,
    history_dir: Path,
    tag: str,
) -> Path:
    """Copy calibration to versioned history archive."""
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_path = history_dir / f"edge_calibration_{tag}.json"
    shutil.copy(calib_path, archive_path)
    logger.info(f"Archived to {archive_path}")
    return archive_path


def _step_publish_latest(
    calib_path: Path,
    latest_path: Path,
) -> None:
    """Promote calibration to runtime-loaded location."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(calib_path, latest_path)
    logger.info(f"Published to {latest_path}")


# ──────────────────────────────────────────────────────────────
# Pipeline orchestration
# ──────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline. Returns exit code."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = args.output_dir / "history"

    tag = pd.Timestamp.now().strftime("%Y-%m")
    start, train_end, end = _resolve_dates(args.years_back, args.oos_days)
    logger.info(
        f"Calibration pipeline start. tag={tag}, "
        f"period={start} ~ {end}, train_end={train_end}"
    )

    # Paths
    panel_path = args.output_dir / f"edge_dataset_{tag}.parquet"
    calib_temp_path = args.output_dir / f"edge_calibration_{tag}.json"
    report_path = (
        args.report_dir / f"validation_{tag}.md"
        if args.report_dir
        else args.output_dir / f"validation_{tag}.md"
    )
    latest_calib_path = args.latest_calibration_path

    # Step 1: Build panel
    cfg = BuildConfig(
        start=start,
        end=end,
        output_path=panel_path,
    )
    ok, err = _step_build(args, cfg, panel_path)
    if not ok:
        logger.error(f"Build step failed: {err}")
        return 2

    # Step 2: Calibrate
    ok, err = _step_calibrate(
        panel_path, calib_temp_path, train_end, args.min_bucket_n
    )
    if not ok:
        logger.error(f"Calibrate step failed: {err}")
        return 2

    # Step 3: Validate
    ok, passed, err = _step_validate(
        panel_path, calib_temp_path, train_end, report_path
    )
    if not ok:
        logger.error(f"Validate step failed: {err}")
        return 2

    # Step 4: Archive (always — keep history even if validation failed)
    _step_archive(calib_temp_path, history_dir, tag)

    # Step 5: Publish to latest (only if validation passed)
    if passed:
        _step_publish_latest(calib_temp_path, latest_calib_path)
        logger.info(f"Pipeline complete: PASS. Latest calibration updated.")
        return 0
    else:
        logger.warning(
            f"Pipeline complete: FAIL. Latest calibration NOT updated. "
            f"Review {report_path}"
        )
        return 1


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ohlcv-path", type=Path, required=True,
        help="OHLCV parquet (long format with date/ticker/...)",
    )
    p.add_argument(
        "--macro-path", type=Path, required=True,
        help="Macro percentile parquet (date-indexed)",
    )
    p.add_argument(
        "--vol-pred-path", type=Path, required=True,
        help="vol_predictions parquet [date, ticker, vol_score, confidence]",
    )
    p.add_argument(
        "--years-back", type=int, default=5,
        help="Lookback years for panel (default 5)",
    )
    p.add_argument(
        "--oos-days", type=int, default=180,
        help="Hold-out days for OOS validation (default 180)",
    )
    p.add_argument(
        "--min-bucket-n", type=int, default=30,
        help="Minimum bucket sample (default 30)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("data/research"),
        help="Pipeline working directory + history",
    )
    p.add_argument(
        "--report-dir", type=Path,
        default=Path("research/reports"),
        help="Markdown validation reports",
    )
    p.add_argument(
        "--latest-calibration-path", type=Path,
        default=Path("v3/config/edge_calibration.json"),
        help="Runtime-loaded calibration JSON (overwritten on PASS)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
