"""Cross-asset regime engine — composite risk-on score with hysteresis.

Philosophy (Bridgewater/AQR hybrid):
  - No single-market indicator (RegimeDetector) → multi-asset composite
  - Rolling 5y percentile, not hardcoded thresholds
  - Hysteresis to prevent whipsaw (2-day confirmation)
  - Continuous [0,1] score → discrete 5-state mapping + scale_factor

States:
  strong_bull:   score >= 0.75   scale 1.2   entry thresholds relaxed
  bull:          score >= 0.55   scale 1.0
  neutral:       score >= 0.40   scale 0.8
  caution:       score >= 0.25   scale 0.4
  bear:          score < 0.25    scale 0.0 (CASH)

Output is RegimeState-compatible so signal.py works unchanged, plus
an extra threshold_multiplier used by EntryFilter for state-adaptive entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from v3.strategy.regime import RegimeState


# Feature weights for composite risk-on score.
# Positive = risk-on contributor; negative = risk-off (inverted).
# Sum of absolute weights = 1.0 (normalized).
FEATURE_WEIGHTS = {
    "breadth":          +0.20,   # More tickers above SMA50 = bullish
    "hyg_tlt_mom_60d":  +0.20,   # Credit appetite = risk-on
    "vix_ratio":        -0.15,   # Inverted: low ratio = complacent = bullish
    "hy_level":         -0.15,   # Inverted: low HY spread = bullish
    "yc_slope":         +0.10,   # Steeper curve = growth expectations
    "dxy_mom_60d":      -0.10,   # Inverted: weaker dollar = risk-on
    "gold_spy_mom_60d": -0.10,   # Inverted: gold underperforming = risk-on
}


@dataclass
class CrossAssetRegimeState(RegimeState):
    score: float = 0.5
    threshold_multiplier: float = 1.0
    contributions: dict = field(default_factory=dict)


class CrossAssetRegimeDetector:
    """Composite cross-asset regime detector with hysteresis."""

    # (lower_bound, regime_name, scale_factor, threshold_multiplier)
    # threshold_multiplier: <1.0 = relax entry filter in bull; >1.0 = tighten in caution
    REGIME_TABLE = [
        (0.75, "strong_bull", 1.2, 0.6),
        (0.55, "bull",        1.0, 0.8),
        (0.40, "neutral",     0.8, 1.0),
        (0.25, "caution",     0.4, 1.3),
        (0.00, "bear",        0.0, 2.0),
    ]

    def __init__(self, hysteresis_days: int = 2):
        self.hysteresis_days = hysteresis_days
        self._current_regime: str = "neutral"
        self._transition_count: int = 0
        self._days_in_regime: int = 0

    def detect(
        self,
        feature_pctl: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> CrossAssetRegimeState:
        """Compute composite regime.

        Args:
            feature_pctl: DataFrame of percentile-ranked features (from MacroFeatureEngineer).
            as_of: Timestamp to evaluate regime at. If None, uses last row.

        Returns:
            CrossAssetRegimeState with regime, score, scale_factor, contributions.
        """
        if feature_pctl is None or feature_pctl.empty:
            return self._fallback_state(reason="empty_features")

        if as_of is None:
            row = feature_pctl.iloc[-1]
        else:
            idx = feature_pctl.index.searchsorted(as_of, side="right") - 1
            if idx < 0:
                return self._fallback_state(reason=f"date {as_of} before data")
            row = feature_pctl.iloc[idx]

        # Composite score from available features
        score, contributions = self._composite_score(row)

        # Map score to regime
        raw_regime, scale, thresh_mult = self._score_to_regime(score)

        # Hysteresis
        confirmed, confidence = self._apply_hysteresis(raw_regime)
        final_scale = scale if confirmed == raw_regime else self._regime_scale(confirmed)
        final_thresh = thresh_mult if confirmed == raw_regime else self._regime_threshold(confirmed)

        return CrossAssetRegimeState(
            regime=confirmed,
            momentum=float(row.get("hyg_tlt_mom_60d", np.nan)) if not pd.isna(row.get("hyg_tlt_mom_60d", np.nan)) else 0.0,
            volatility=float(row.get("vix_ratio", np.nan)) if not pd.isna(row.get("vix_ratio", np.nan)) else 0.0,
            scale_factor=final_scale,
            confidence=confidence,
            days_in_regime=self._days_in_regime,
            score=float(score),
            threshold_multiplier=final_thresh,
            contributions=contributions,
        )

    def _composite_score(self, row: pd.Series) -> tuple[float, dict]:
        """Weighted sum of percentiles → [0, 1] risk-on score."""
        numerator = 0.0
        total_weight = 0.0
        contribs: dict = {}

        for feature, weight in FEATURE_WEIGHTS.items():
            if feature not in row.index:
                continue
            pctl = row[feature]
            if pd.isna(pctl):
                continue

            # Positive weight: percentile itself contributes.
            # Negative weight: inverted (1 - percentile).
            if weight > 0:
                contribution = pctl
            else:
                contribution = 1.0 - pctl

            numerator += abs(weight) * contribution
            total_weight += abs(weight)
            contribs[feature] = round(float(contribution * abs(weight)), 4)

        if total_weight == 0:
            return 0.5, contribs

        score = numerator / total_weight
        return float(np.clip(score, 0.0, 1.0)), contribs

    def _score_to_regime(self, score: float) -> tuple[str, float, float]:
        """Map score to (regime_name, scale_factor, threshold_multiplier)."""
        for lower, name, scale, thresh in self.REGIME_TABLE:
            if score >= lower:
                return name, scale, thresh
        return "bear", 0.0, 2.0

    def _regime_scale(self, regime: str) -> float:
        for _, name, scale, _ in self.REGIME_TABLE:
            if name == regime:
                return scale
        return 1.0

    def _regime_threshold(self, regime: str) -> float:
        for _, name, _, thresh in self.REGIME_TABLE:
            if name == regime:
                return thresh
        return 1.0

    def _apply_hysteresis(self, raw_regime: str) -> tuple[str, float]:
        """Require N consecutive days in new regime before switching."""
        if raw_regime != self._current_regime:
            self._transition_count += 1
            if self._transition_count >= self.hysteresis_days:
                self._current_regime = raw_regime
                self._transition_count = 0
                self._days_in_regime = 1
                return raw_regime, 1.0
            else:
                confidence = 1.0 - (self._transition_count / self.hysteresis_days) * 0.3
                return self._current_regime, confidence
        else:
            self._transition_count = 0
            self._days_in_regime += 1
            return self._current_regime, 1.0

    def _fallback_state(self, reason: str) -> CrossAssetRegimeState:
        """Return neutral state when data unavailable."""
        logger.warning(f"CrossAssetRegime fallback: {reason}")
        return CrossAssetRegimeState(
            regime="neutral",
            momentum=0.0,
            volatility=0.0,
            scale_factor=0.8,
            confidence=0.5,
            days_in_regime=0,
            score=0.5,
            threshold_multiplier=1.0,
            contributions={},
        )
