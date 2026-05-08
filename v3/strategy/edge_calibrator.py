"""V3.3 EdgeCalibrator — raw_opportunity → expected_return_5d.

Loads calibration_table.json (built by v3/research/calibrate_edge.py from
the panel produced in PR-1.4) and applies bucket lookup with fallback chain.

Fallback chain (Level 4 sector deliberately excluded — sample 부족):
  1. (decile, regime, liquidity_bucket)   → ~150 buckets, ~99 row each
  2. (decile, regime)                     → ~50 buckets, ~296 row each
  3. (decile,)                            → 10 buckets, ~1482 row each
  4. ("global",)                          → 1 bucket, full panel

When sample at all levels < min_bucket_n:
  insufficient_action == "block":  reject (entry_pass=False)
  insufficient_action == "global": fall back to global mean (lossy)
  insufficient_action == "demote": apply 0.5× shrinkage to global mean

Pure function design — no mutation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
from loguru import logger

from v3.strategy.types import CalibrationBucket, EdgeCandidate


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DECILE_COUNT = 10
DEFAULT_MIN_BUCKET_N = 30


# ──────────────────────────────────────────────────────────────
# Calibration table I/O
# ──────────────────────────────────────────────────────────────
def serialize_bucket_key(key: tuple) -> str:
    """Tuple key → string for JSON storage."""
    return "|".join(str(x) if x is not None else "_NONE_" for x in key)


def deserialize_bucket_key(s: str) -> tuple:
    """String → tuple key."""
    parts = s.split("|")
    return tuple(None if p == "_NONE_" else p for p in parts)


def save_calibration_table(
    table: dict[tuple, CalibrationBucket],
    path: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    """Persist calibration table as JSON.

    Args:
        table: {bucket_key_tuple: CalibrationBucket}
        path: output JSON path
        metadata: optional fields like {"version": "...", "panel_size": ..., ...}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "buckets": {},
    }
    for key, bucket in table.items():
        payload["buckets"][serialize_bucket_key(key)] = {
            "key": list(key),
            "n": bucket.n,
            "mean_forward_return_5d": bucket.mean_forward_return_5d,
            "mean_mae_5d": bucket.mean_mae_5d,
            "mean_mfe_5d": bucket.mean_mfe_5d,
            "win_rate_5d": bucket.win_rate_5d,
            "std_forward_return_5d": bucket.std_forward_return_5d,
            "median_forward_return_5d": bucket.median_forward_return_5d,
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def load_calibration_table(
    path: str | Path,
) -> tuple[dict[tuple, CalibrationBucket], dict]:
    """Load calibration table from JSON.

    Returns:
        (table, metadata)
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {})
    table: dict[tuple, CalibrationBucket] = {}

    for key_str, bucket_dict in payload.get("buckets", {}).items():
        key = deserialize_bucket_key(key_str)
        table[key] = CalibrationBucket(
            key=key,
            n=int(bucket_dict["n"]),
            mean_forward_return_5d=float(bucket_dict["mean_forward_return_5d"]),
            mean_mae_5d=float(bucket_dict["mean_mae_5d"]),
            mean_mfe_5d=float(bucket_dict["mean_mfe_5d"]),
            win_rate_5d=float(bucket_dict["win_rate_5d"]),
            std_forward_return_5d=float(bucket_dict["std_forward_return_5d"]),
            median_forward_return_5d=float(bucket_dict["median_forward_return_5d"]),
        )

    return table, metadata


# ──────────────────────────────────────────────────────────────
# Decile assignment
# ──────────────────────────────────────────────────────────────
def assign_decile(
    raw_opportunity: float,
    decile_breakpoints: list[float],
) -> int:
    """Map raw_opportunity to decile index [0, 9].

    decile_breakpoints: sorted list of 9 cut points (10th, 20th, ..., 90th pct).
    """
    if not decile_breakpoints or len(decile_breakpoints) != DECILE_COUNT - 1:
        raise ValueError(
            f"decile_breakpoints must have {DECILE_COUNT - 1} values, "
            f"got {len(decile_breakpoints) if decile_breakpoints else 0}"
        )
    for i, bp in enumerate(decile_breakpoints):
        if raw_opportunity < bp:
            return i
    return DECILE_COUNT - 1


# ──────────────────────────────────────────────────────────────
# EdgeCalibrator
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for EdgeCalibrator behavior."""
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N
    insufficient_action: Literal["block", "global", "demote"] = "block"
    demote_factor: float = 0.5


class EdgeCalibrator:
    """raw_opportunity → expected_return_5d via bucket lookup with fallback.

    Pure function design. Stateless except for loaded calibration table.

    Usage:
        cal = EdgeCalibrator.from_file(
            "v3/config/edge_calibration.json",
            min_bucket_n=30,
        )
        enriched = cal.calibrate(candidate)
        # or batch:
        enriched_list = cal.calibrate_batch(candidates)
    """

    def __init__(
        self,
        table: dict[tuple, CalibrationBucket],
        decile_breakpoints: list[float],
        config: Optional[CalibrationConfig] = None,
        metadata: Optional[dict] = None,
    ):
        self.table = table
        self.decile_breakpoints = decile_breakpoints
        self.config = config or CalibrationConfig()
        self.metadata = metadata or {}

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        config: Optional[CalibrationConfig] = None,
    ) -> "EdgeCalibrator":
        """Load from calibration_table.json."""
        table, metadata = load_calibration_table(path)
        decile_bp = metadata.get("decile_breakpoints")
        if decile_bp is None:
            raise ValueError(
                f"Calibration table at {path} missing decile_breakpoints in metadata"
            )
        return cls(
            table=table,
            decile_breakpoints=list(decile_bp),
            config=config,
            metadata=metadata,
        )

    # ── public API ────────────────────────────────────────────
    def calibrate(self, candidate: EdgeCandidate) -> EdgeCandidate:
        """Apply calibration to a single candidate. Returns new instance."""
        decile = assign_decile(candidate.raw_opportunity, self.decile_breakpoints)
        keys = self._fallback_keys(decile, candidate.regime, candidate.liquidity_bucket)

        for key in keys:
            bucket = self.table.get(key)
            if bucket is not None and bucket.n >= self.config.min_bucket_n:
                return candidate.with_calibration(
                    expected_return_5d=bucket.mean_forward_return_5d,
                    expected_mae_5d=bucket.mean_mae_5d,
                    expected_mfe_5d=bucket.mean_mfe_5d,
                    win_probability_5d=bucket.win_rate_5d,
                    calibration_n=bucket.n,
                    calibration_key=key,
                )

        return self._handle_insufficient(candidate)

    def calibrate_batch(
        self,
        candidates: Iterable[EdgeCandidate],
    ) -> list[EdgeCandidate]:
        """Apply calibration to multiple candidates. List of new instances."""
        return [self.calibrate(c) for c in candidates]

    # ── internal ──────────────────────────────────────────────
    def _fallback_keys(
        self,
        decile: int,
        regime: str,
        liquidity_bucket: Optional[str],
    ) -> list[tuple]:
        """Fallback chain — decile-based lookup only.

        Global bucket is reserved for _handle_insufficient (last-resort path
        controlled by insufficient_action policy). Including it here would
        bypass the policy gate.

        Level 4 (sector) deliberately excluded — sample 부족.
        """
        keys: list[tuple] = []
        if liquidity_bucket:
            keys.append((decile, regime, liquidity_bucket))
        keys.append((decile, regime))
        keys.append((decile,))
        return keys

    def _handle_insufficient(self, candidate: EdgeCandidate) -> EdgeCandidate:
        """Apply insufficient_action policy."""
        action = self.config.insufficient_action

        if action == "block":
            return candidate.with_calibration(
                entry_pass=False,
                reject_reason="insufficient_calibration",
            )

        # "global" or "demote" — fall back to global if exists
        global_bucket = self.table.get(("global",))
        if global_bucket is None or global_bucket.n < self.config.min_bucket_n:
            # Even global insufficient — block
            return candidate.with_calibration(
                entry_pass=False,
                reject_reason="insufficient_calibration",
            )

        factor = self.config.demote_factor if action == "demote" else 1.0
        return candidate.with_calibration(
            expected_return_5d=global_bucket.mean_forward_return_5d * factor,
            expected_mae_5d=global_bucket.mean_mae_5d,
            expected_mfe_5d=global_bucket.mean_mfe_5d,
            win_probability_5d=global_bucket.win_rate_5d,
            calibration_n=global_bucket.n,
            calibration_key=("global",),
        )
