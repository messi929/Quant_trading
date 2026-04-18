"""Regime detector V2 — frozen dataclass, loads learned alpha_weights (Phase 2 S3).

Replaces v3/strategy/regime.py (single-asset) and v3/strategy/regime_cross_asset.py
(patch-style). Single canonical detector for Phase 2.

Design:
  1. Frozen dataclass Regime (immutable snapshot)
  2. Regime = 5 discrete states + continuous score
  3. alpha_weights loaded from v3/config/alpha_weights.json (S2 output)
  4. position_scale is a smooth function of score (continuous)
  5. Hysteresis prevents whipsaw transitions

Data flow:
  MacroCollector + MacroFeatureEngineer  →  feature percentiles
                                             ↓
                                        composite_score [0, 1]
                                             ↓
                                   hysteresis + quantile thresholds
                                             ↓
                                     Regime(name, score,
                                            alpha_weights, position_scale)

Output is consumed by:
  - opportunity.py (S4): uses alpha_weights to combine directional alphas
  - signal.py (S4): uses position_scale for sizing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# Must match REGIME_THRESHOLDS in alpha_weight_trainer.py
REGIME_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "strong_bull"),
    (0.55, "bull"),
    (0.40, "neutral"),
    (0.25, "caution"),
    (0.00, "bear"),
]
REGIME_NAMES: list[str] = [name for _, name in REGIME_THRESHOLDS]

# Position scale curve: smooth function of score [0, 1] → [0, 1.2]
# - score 0.00 → 0.00 (bear = cash)
# - score 0.25 → 0.30
# - score 0.40 → 0.60
# - score 0.55 → 0.90
# - score 0.75 → 1.10
# - score 1.00 → 1.20
POSITION_SCALE_CURVE: list[tuple[float, float]] = [
    (0.00, 0.00),
    (0.25, 0.30),
    (0.40, 0.60),
    (0.55, 0.90),
    (0.75, 1.10),
    (1.00, 1.20),
]


# ──────────────────────────────────────────────────────────────
# Regime value object (immutable)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Regime:
    """Immutable regime snapshot consumed by opportunity scorer and sizer.

    Attributes:
        name: One of REGIME_NAMES (strong_bull | bull | neutral | caution | bear).
        score: Continuous composite score in [0, 1].
        alpha_weights: dict[alpha_name, weight], sum ≈ 1.0.
        position_scale: Continuous scale factor in [0, 1.2].
        confidence: Hysteresis-based confidence in [0.7, 1.0].
        detected_at: Timestamp of detection.
        feature_contributions: For observability only, not used downstream.
    """

    name: str
    score: float
    alpha_weights: dict[str, float]
    position_scale: float
    confidence: float
    detected_at: pd.Timestamp
    feature_contributions: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        """Short human-readable summary for logs."""
        weights = ", ".join(f"{a}={w:.2f}" for a, w in self.alpha_weights.items())
        return (
            f"Regime({self.name} score={self.score:.2f} "
            f"scale={self.position_scale:.2f} conf={self.confidence:.2f} "
            f"weights=[{weights}])"
        )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _score_to_name(score: float) -> str:
    """Map [0, 1] score to regime name."""
    for lower, name in REGIME_THRESHOLDS:
        if score >= lower:
            return name
    return "bear"


def _score_to_position_scale(score: float) -> float:
    """Piecewise-linear interpolation over POSITION_SCALE_CURVE."""
    score = float(np.clip(score, 0.0, 1.0))
    for i in range(len(POSITION_SCALE_CURVE) - 1):
        x0, y0 = POSITION_SCALE_CURVE[i]
        x1, y1 = POSITION_SCALE_CURVE[i + 1]
        if x0 <= score <= x1:
            if x1 - x0 < 1e-9:
                return y0
            t = (score - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return POSITION_SCALE_CURVE[-1][1]


# ──────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────
class RegimeDetectorV2:
    """Composite cross-asset regime detector with hysteresis.

    Produces immutable Regime snapshots. Loads learned alpha_weights.json from
    S2 trainer output. If weights file missing, uses uniform weights (fallback).
    """

    def __init__(
        self,
        weights_path: str | Path = "v3/config/alpha_weights.json",
        hysteresis_days: int = 2,
        feature_weights: dict[str, float] | None = None,
        feature_signs: dict[str, float] | None = None,
    ):
        """Initialize detector.

        Args:
            weights_path: Path to alpha_weights.json (from S2 trainer).
            hysteresis_days: Days to confirm a regime transition.
            feature_weights: Optional override for macro feature weights in
                composite score. Defaults to uniform over all features present.
            feature_signs: Optional override for feature signs (+1 risk-on,
                -1 risk-off). Defaults to +1 for all.
        """
        self.weights_path = Path(weights_path)
        self.hysteresis_days = hysteresis_days

        # Load learned alpha weights
        self.artifact = self._load_artifact(self.weights_path)
        self.alpha_weights_by_regime: dict[str, dict[str, float]] = (
            self.artifact.get("directional_weights", {})
        )
        self.directional_alpha_names: list[str] = (
            self.artifact.get("directional_alphas", [])
        )
        self._fill_missing_regime_weights()

        # Feature weights / signs: explicit override > artifact regime_composite
        rc = self.artifact.get("regime_composite", {})
        self.feature_weights = feature_weights or rc.get("feature_weights") or None
        self.feature_signs = feature_signs or rc.get("feature_signs") or None

        # Hysteresis state
        self._current_regime: str = "neutral"
        self._transition_count: int = 0
        self._days_in_regime: int = 0

    # ── public API ────────────────────────────────────────
    def detect(
        self,
        feature_pctl: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> Regime:
        """Compute regime from macro feature percentiles.

        Args:
            feature_pctl: Rolling percentile of macro features (from
                MacroFeatureEngineer.compute_percentiles).
            as_of: Target timestamp. If None, uses last row of feature_pctl.

        Returns:
            Immutable Regime snapshot.
        """
        if feature_pctl is None or feature_pctl.empty:
            return self._fallback_regime(
                reason="empty feature percentiles",
                as_of=as_of or pd.Timestamp(datetime.now().date()),
            )

        if as_of is None:
            row = feature_pctl.iloc[-1]
            as_of = feature_pctl.index[-1]
        else:
            idx = feature_pctl.index.searchsorted(as_of, side="right") - 1
            if idx < 0:
                return self._fallback_regime(
                    reason=f"date {as_of} before data",
                    as_of=as_of,
                )
            row = feature_pctl.iloc[idx]

        # Composite score from macro features
        score, contributions = self._composite_score(row)

        # Raw regime name from score
        raw_name = _score_to_name(score)

        # Hysteresis → confirmed regime name
        confirmed_name, confidence = self._apply_hysteresis(raw_name)

        # Scale is based on SCORE (continuous), not regime bucket → smooth
        position_scale = _score_to_position_scale(score)

        # Alpha weights for confirmed regime (dict copy for immutability)
        weights = dict(self.alpha_weights_by_regime.get(confirmed_name, {}))

        return Regime(
            name=confirmed_name,
            score=float(score),
            alpha_weights=weights,
            position_scale=float(position_scale),
            confidence=float(confidence),
            detected_at=pd.Timestamp(as_of),
            feature_contributions=contributions,
        )

    def warm_up(self, feature_pctl: pd.DataFrame, days: int = 5) -> None:
        """Run detect() on last N days (in order) to stabilize hysteresis.

        Use at service startup when detector state is fresh.
        """
        if feature_pctl is None or len(feature_pctl) < 2:
            return
        tail = feature_pctl.tail(days)
        for ts in tail.index:
            self.detect(feature_pctl, as_of=ts)

    # ── internal ──────────────────────────────────────────
    def _composite_score(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        """Weighted average of feature percentiles (or inverse where sign=-1)."""
        if self.feature_weights is None:
            # Uniform weights over features present in this row
            feats = [c for c in row.index if pd.notna(row[c])]
            if not feats:
                return 0.5, {}
            w = 1.0 / len(feats)
            weights = {c: w for c in feats}
        else:
            weights = self.feature_weights

        signs = self.feature_signs or {}

        num = 0.0
        denom = 0.0
        contribs: dict[str, float] = {}
        for feat, w in weights.items():
            if feat not in row.index:
                continue
            pctl = row[feat]
            if pd.isna(pctl):
                continue
            sign = signs.get(feat, 1.0)
            contribution = float(pctl) if sign >= 0 else (1.0 - float(pctl))
            num += w * contribution
            denom += w
            contribs[feat] = round(w * contribution, 4)

        if denom <= 1e-9:
            return 0.5, contribs
        return float(np.clip(num / denom, 0.0, 1.0)), contribs

    def _apply_hysteresis(self, raw_regime: str) -> tuple[str, float]:
        if raw_regime != self._current_regime:
            self._transition_count += 1
            if self._transition_count >= self.hysteresis_days:
                self._current_regime = raw_regime
                self._transition_count = 0
                self._days_in_regime = 1
                return raw_regime, 1.0
            # Partial confidence while in transition
            confidence = 1.0 - (self._transition_count / self.hysteresis_days) * 0.3
            return self._current_regime, float(confidence)
        self._transition_count = 0
        self._days_in_regime += 1
        return self._current_regime, 1.0

    def _load_artifact(self, path: Path) -> dict:
        if not path.exists():
            logger.warning(
                f"alpha_weights.json not found at {path} — run "
                "`python v3/backtest/alpha_weight_trainer.py` first. "
                "Using empty fallback (uniform weights at detect time)."
            )
            return {}
        with path.open("r", encoding="utf-8") as f:
            art = json.load(f)
        logger.info(
            f"Loaded alpha_weights.json: generated_at={art.get('generated_at')}, "
            f"regimes={list(art.get('directional_weights', {}).keys())}, "
            f"alphas={art.get('directional_alphas')}"
        )
        return art

    def _fill_missing_regime_weights(self) -> None:
        """Ensure every regime has a weights dict; fallback to uniform."""
        alpha_names = self.directional_alpha_names
        if not alpha_names:
            return
        n = len(alpha_names)
        uniform = {a: round(1.0 / n, 4) for a in alpha_names}
        for regime_name in REGIME_NAMES:
            if regime_name not in self.alpha_weights_by_regime:
                self.alpha_weights_by_regime[regime_name] = dict(uniform)

    def _fallback_regime(self, reason: str, as_of: pd.Timestamp) -> Regime:
        logger.warning(f"RegimeDetectorV2 fallback: {reason}")
        alpha_names = self.directional_alpha_names or []
        uniform = (
            {a: round(1.0 / len(alpha_names), 4) for a in alpha_names}
            if alpha_names else {}
        )
        return Regime(
            name="neutral",
            score=0.5,
            alpha_weights=uniform,
            position_scale=_score_to_position_scale(0.5),
            confidence=0.5,
            detected_at=as_of,
            feature_contributions={},
        )


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build a tiny synthetic feature_pctl and test detector
    dates = pd.date_range("2026-01-01", periods=30, freq="B")

    # Simulate gradually rising risk-on
    scores = np.linspace(0.30, 0.80, len(dates))
    feature_pctl = pd.DataFrame({
        "breadth":  np.clip(scores + np.random.normal(0, 0.05, len(dates)), 0, 1),
        "vix_ratio": np.clip(1 - scores + np.random.normal(0, 0.05, len(dates)), 0, 1),
        "hy_level":  np.clip(1 - scores + np.random.normal(0, 0.05, len(dates)), 0, 1),
    }, index=dates)

    det = RegimeDetectorV2(
        feature_weights={"breadth": 0.5, "vix_ratio": 0.25, "hy_level": 0.25},
        feature_signs={"breadth": 1.0, "vix_ratio": -1.0, "hy_level": -1.0},
    )

    print("=== Smoke test: rising risk-on ===")
    for ts in dates[::5]:
        r = det.detect(feature_pctl, as_of=ts)
        print(f"  {ts.date()}  {r.describe()}")
