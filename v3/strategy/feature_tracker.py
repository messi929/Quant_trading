"""V3.3 feature activation tracker (F2 fix).

Detects feature flag transitions (false → true) on startup and records
to feature_activations.jsonl for rollback consumption.

Wiring:
  - live_pipeline.__init__ calls record_features_on_startup() once per session
  - rollback_check.py reads the JSONL via load_activations()

Without this hook the rollback safety net is silently broken — Evaluator
flagged this as a P0 gap.

Mechanism:
  feature_state_snapshot.json stores the last-seen features dict.
  On each startup, current features (from V3Config.features) are compared
  to the snapshot. Each false→true transition emits an activation event.
  Snapshot is then refreshed.

Manual override:
  v3/scripts/record_activation.py allows operator to record a single
  activation explicitly (when deploying via different mechanism).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from v3.config.schema import FeatureFlagsConfig
from v3.research.rollback import append_activation


# ──────────────────────────────────────────────────────────────
# Snapshot I/O
# ──────────────────────────────────────────────────────────────
def _load_snapshot(path: Path) -> dict:
    """Returns {} if missing or unreadable."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_snapshot(
    path: Path,
    features_dict: dict[str, bool],
    timestamp: pd.Timestamp,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_seen": timestamp.isoformat(),
        "features": features_dict,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ──────────────────────────────────────────────────────────────
# Detect transitions
# ──────────────────────────────────────────────────────────────
def detect_newly_activated(
    current: FeatureFlagsConfig,
    snapshot: dict,
) -> list[str]:
    """Compare current features dict to previous snapshot.

    Returns sorted list of features that transitioned false → true.
    """
    current_dict = current.model_dump()
    prev_features = snapshot.get("features", {})

    newly_active: list[str] = []
    for feature, current_val in current_dict.items():
        prev_val = prev_features.get(feature, False)
        if current_val and not prev_val:
            newly_active.append(feature)

    return sorted(newly_active)


# ──────────────────────────────────────────────────────────────
# Public API — called from live_pipeline startup
# ──────────────────────────────────────────────────────────────
def record_features_on_startup(
    current: FeatureFlagsConfig,
    history_path: Path,
    snapshot_path: Path,
    activated_by: str = "auto",
    now: Optional[pd.Timestamp] = None,
) -> int:
    """Detect feature toggles and append to history JSONL.

    Returns number of newly recorded activations.

    Side effects:
      - Appends to history_path (one JSON line per new activation)
      - Overwrites snapshot_path with current features

    Idempotent — safe to call on every startup. If no toggles, no events
    emitted, snapshot is refreshed only with new timestamp.
    """
    now = now if now is not None else pd.Timestamp.now()
    snapshot = _load_snapshot(snapshot_path)
    newly_active = detect_newly_activated(current, snapshot)

    for feature in newly_active:
        append_activation(
            feature=feature,
            activated_at=now,
            history_path=history_path,
            activated_by=activated_by,
        )
        logger.info(
            f"feature_tracker: recorded activation '{feature}' "
            f"(by={activated_by})"
        )

    # Always refresh snapshot — keeps timestamp current
    _save_snapshot(snapshot_path, current.model_dump(), now)

    if newly_active:
        logger.warning(
            f"feature_tracker: {len(newly_active)} new activation(s) "
            f"recorded — rollback safety net updated"
        )

    return len(newly_active)
