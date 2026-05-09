"""V3.3 daily rollback check (P7).

Production cron entry (KST 16:30 after daily report):
  30 16 * * * /opt/quant/run_rollback_check.sh

Runs:
  1. Load paper_account.json → 7d realized PnL
  2. Load feature_activations.jsonl → activation history
  3. Evaluate rollback (threshold default -2%)
  4. If triggered: edit v3_config.yaml (set features.<X>=false)
  5. Append rollback_history.jsonl + write alert markdown
  6. (optional) systemctl restart quant-trading-v3 — manual hook

Exit codes:
  0  no rollback (PnL ok)
  1  rollback triggered + applied
  2  pipeline error (paper_account missing, yaml structure invalid, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from v3.research.rollback import (
    append_rollback_log,
    apply_rollback_to_yaml,
    compute_window_pnl_pct,
    evaluate_rollback,
    load_activations,
    render_rollback_alert,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paper-account", type=Path,
        default=Path("v3/saved_models/paper_account.json"),
        help="paper account JSON (trade_history)",
    )
    p.add_argument(
        "--config-path", type=Path,
        default=Path("v3/config/v3_config.yaml"),
        help="V3 config YAML to mutate on rollback",
    )
    p.add_argument(
        "--activation-history", type=Path,
        default=Path("v3/saved_models/feature_activations.jsonl"),
        help="JSONL of feature activation events",
    )
    p.add_argument(
        "--log-path", type=Path,
        default=Path("v3/saved_models/rollback_history.jsonl"),
        help="JSONL append target for rollback decisions",
    )
    p.add_argument(
        "--alert-dir", type=Path,
        default=Path("research/reports/rollback"),
        help="Directory for rollback alert markdown",
    )
    p.add_argument(
        "--window-days", type=int, default=7,
    )
    p.add_argument(
        "--threshold", type=float, default=-0.02,
        help="Rollback when pnl_window_pct < threshold (default -0.02 = -2%)",
    )
    p.add_argument(
        "--base-capital", type=float, default=100_000_000.0,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate but do NOT modify yaml or log",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    today = pd.Timestamp.now().normalize()

    if not args.paper_account.exists():
        logger.error(f"paper_account not found: {args.paper_account}")
        return 2

    with args.paper_account.open("r", encoding="utf-8") as f:
        account = json.load(f)
    trades = account.get("trade_history", [])

    pnl_pct = compute_window_pnl_pct(
        trades, today, args.window_days, args.base_capital,
    )
    activations = load_activations(args.activation_history)

    decision = evaluate_rollback(
        pnl_window_pct=pnl_pct,
        activations=activations,
        threshold_pct=args.threshold,
        window_days=args.window_days,
    )

    logger.info(
        f"Rollback check: pnl_{args.window_days}d={pnl_pct:+.4f} "
        f"(threshold {args.threshold:+.4f}). triggered={decision.triggered}"
    )

    # Always write alert markdown
    alert_path = args.alert_dir / f"{today.strftime('%Y-%m-%d')}.md"
    if not args.dry_run:
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        alert_path.write_text(
            render_rollback_alert(decision), encoding="utf-8"
        )

    if not decision.triggered:
        if not args.dry_run:
            append_rollback_log(decision, args.log_path)
        return 0

    # Triggered — apply
    if args.dry_run:
        logger.warning(
            f"DRY RUN: would rollback {decision.feature_to_rollback}"
        )
        return 1

    ok = apply_rollback_to_yaml(args.config_path, decision.feature_to_rollback)
    append_rollback_log(decision, args.log_path)

    if ok:
        logger.warning(
            f"ROLLBACK APPLIED: features.{decision.feature_to_rollback} → false"
        )
        logger.warning(
            "NOTE: systemctl restart quant-trading-v3 required to take effect"
        )
        return 1

    logger.error(
        f"Rollback decision but apply_rollback_to_yaml failed for "
        f"{decision.feature_to_rollback}"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
