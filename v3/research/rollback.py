"""V3.3 automatic feature rollback (P7).

Detects PnL degradation after feature activation and rolls back the most
recent activated feature.

Decision logic:
  1. Sum realized PnL over `window_days` (default 7) from paper_account
  2. If pnl_window_pct < threshold (default -2%) AND there are activations:
     → rollback the most recently activated feature
  3. Update v3_config.yaml `features.<X>` to false
  4. Log decision to rollback_history.jsonl

Wiring:
  - Daily cron (systemd timer): 16:30 KST after daily report
  - Manual override: `python v3/scripts/run_rollback_check.py --dry-run`

Invariants:
  - Diagnostic features (no_trade_logger / tc_monitor / execution_quality)
    are NEVER rolled back (read-only, not the cause of PnL change)
  - rollback never re-activates a previously OFF feature
  - apply_rollback returns False if YAML structure unexpected
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger


# Diagnostic features cannot be rolled back (read-only)
NEVER_ROLLBACK: frozenset[str] = frozenset({
    "no_trade_logger",
    "tc_monitor",
    "execution_quality",
})


# ──────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureActivation:
    """A single feature activation event."""
    feature: str
    activated_at: pd.Timestamp
    activated_by: str = "manual"


@dataclass(frozen=True)
class RollbackDecision:
    """Result of rollback evaluation."""
    triggered: bool
    reason: str
    feature_to_rollback: Optional[str]
    pnl_window_pct: float
    window_days: int


# ──────────────────────────────────────────────────────────────
# PnL computation
# ──────────────────────────────────────────────────────────────
def compute_window_pnl_pct(
    trade_history: list[dict],
    today: pd.Timestamp,
    window_days: int = 7,
    base_capital: float = 100_000_000.0,
) -> float:
    """Sum realized net_pnl in window / base_capital → fractional pct.

    Trade history field tolerance:
      exit_date / entry_date (timestamp string)
      net_pnl_krw / realized_pnl_krw / pnl_krw (numeric)
    """
    if base_capital <= 0:
        return 0.0
    cutoff = today - pd.Timedelta(days=window_days)
    total = 0.0
    for t in trade_history:
        # Date field
        date_raw = t.get("exit_date") or t.get("entry_date")
        if not date_raw:
            continue
        try:
            ts = pd.Timestamp(date_raw)
        except (ValueError, TypeError):
            continue
        if ts < cutoff or ts > today:
            continue
        # PnL field — try common variants
        pnl_raw = (
            t.get("net_pnl_krw")
            or t.get("realized_pnl_krw")
            or t.get("pnl_krw")
            or 0.0
        )
        try:
            total += float(pnl_raw)
        except (ValueError, TypeError):
            continue
    return total / base_capital


# ──────────────────────────────────────────────────────────────
# Rollback decision
# ──────────────────────────────────────────────────────────────
def evaluate_rollback(
    pnl_window_pct: float,
    activations: list[FeatureActivation],
    threshold_pct: float = -0.02,
    window_days: int = 7,
) -> RollbackDecision:
    """Decide if rollback should fire.

    Returns triggered=False if:
      - pnl above threshold (no degradation)
      - no rollback-eligible activations (all NEVER_ROLLBACK)
    """
    if pnl_window_pct >= threshold_pct:
        return RollbackDecision(
            triggered=False,
            reason=(
                f"pnl_{window_days}d={pnl_window_pct:+.4f} ≥ "
                f"threshold {threshold_pct:+.4f}"
            ),
            feature_to_rollback=None,
            pnl_window_pct=pnl_window_pct,
            window_days=window_days,
        )

    eligible = [a for a in activations if a.feature not in NEVER_ROLLBACK]
    if not eligible:
        return RollbackDecision(
            triggered=False,
            reason=(
                f"pnl_{window_days}d={pnl_window_pct:+.4f} below threshold "
                f"but no rollback-eligible activations "
                f"(diagnostics-only, will not rollback)"
            ),
            feature_to_rollback=None,
            pnl_window_pct=pnl_window_pct,
            window_days=window_days,
        )

    most_recent = max(eligible, key=lambda a: a.activated_at)
    return RollbackDecision(
        triggered=True,
        reason=(
            f"pnl_{window_days}d={pnl_window_pct:+.4f} < {threshold_pct:+.4f} "
            f"→ rollback most recent activation: "
            f"{most_recent.feature} (activated {most_recent.activated_at.date()})"
        ),
        feature_to_rollback=most_recent.feature,
        pnl_window_pct=pnl_window_pct,
        window_days=window_days,
    )


# ──────────────────────────────────────────────────────────────
# YAML mutation
# ──────────────────────────────────────────────────────────────
def apply_rollback_to_yaml(yaml_path: Path, feature: str) -> bool:
    """Set features.<feature> = false in v3_config.yaml.

    Returns False if:
      - YAML missing features section
      - feature not in features
      - feature already false (no-op signal)

    Note: yaml.safe_dump loses comments. v3_config.yaml comments will be
    regenerated as plain YAML. Acceptable trade-off for automation.
    """
    if feature in NEVER_ROLLBACK:
        logger.warning(
            f"Refusing to rollback {feature}: in NEVER_ROLLBACK set"
        )
        return False

    if not yaml_path.exists():
        logger.error(f"Config not found: {yaml_path}")
        return False

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "features" not in cfg or feature not in cfg["features"]:
        logger.error(
            f"features.{feature} not present in {yaml_path}"
        )
        return False

    if not cfg["features"][feature]:
        logger.warning(f"features.{feature} already false — no-op")
        return False

    cfg["features"][feature] = False

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    logger.warning(
        f"ROLLBACK applied: features.{feature} → false in {yaml_path}"
    )
    return True


# ──────────────────────────────────────────────────────────────
# Activation history I/O
# ──────────────────────────────────────────────────────────────
def load_activations(path: Path) -> list[FeatureActivation]:
    """Load FeatureActivation list from JSONL history."""
    if not path.exists():
        return []
    out: list[FeatureActivation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out.append(FeatureActivation(
                    feature=str(r["feature"]),
                    activated_at=pd.Timestamp(r["activated_at"]),
                    activated_by=str(r.get("activated_by", "manual")),
                ))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def append_activation(
    feature: str,
    activated_at: pd.Timestamp,
    history_path: Path,
    activated_by: str = "manual",
) -> None:
    """Append a new activation event to history."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "feature": feature,
            "activated_at": activated_at.isoformat(),
            "activated_by": activated_by,
        }) + "\n")


def append_rollback_log(
    decision: RollbackDecision,
    log_path: Path,
) -> None:
    """Append rollback decision to history JSONL."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": pd.Timestamp.now().isoformat(),
            "triggered": decision.triggered,
            "feature_to_rollback": decision.feature_to_rollback,
            "pnl_window_pct": decision.pnl_window_pct,
            "window_days": decision.window_days,
            "reason": decision.reason,
        }) + "\n")


# ──────────────────────────────────────────────────────────────
# Markdown alert
# ──────────────────────────────────────────────────────────────
def render_rollback_alert(decision: RollbackDecision) -> str:
    """Short markdown alert for daily report or notification."""
    lines = ["# V3.3 Rollback Alert", ""]
    if decision.triggered:
        lines.append(f"**TRIGGERED**: {decision.feature_to_rollback} → false")
    else:
        lines.append("**No rollback**")
    lines.extend([
        f"- pnl_{decision.window_days}d: {decision.pnl_window_pct:+.4f}",
        f"- reason: {decision.reason}",
    ])
    return "\n".join(lines)
