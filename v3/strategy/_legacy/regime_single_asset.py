"""Market regime detection with hysteresis and mean-reversion mode.

Medallion-inspired enhancements:
  1. Hysteresis: require 3 consecutive days to confirm regime switch
  2. Cost-aware regime switching (don't liquidate for marginal bear calls)
  3. Adaptive momentum window (longer in high vol, shorter in low vol)
  4. Mean-reversion regime detection for KRX
  5. Regime confidence scoring
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RegimeState:
    regime: str          # "bull", "bear", "volatile", "neutral", "mean_reversion"
    momentum: float
    volatility: float
    scale_factor: float
    confidence: float = 1.0     # How confident are we in this regime call
    days_in_regime: int = 0     # How long we've been in this regime


class RegimeDetector:
    """Detects market regime with hysteresis to prevent whipsaw switching."""

    SCALE_FACTORS = {
        "bull": 1.2,
        "bear": 0.0,       # CASH
        "volatile": 0.6,
        "neutral": 1.0,
        "mean_reversion": 0.8,
    }

    def __init__(
        self,
        momentum_window: int = 20,
        vol_lookback: int = 60,
        vol_percentile: float = 0.80,
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.08,
        hysteresis_days: int = 3,
    ):
        self.default_window = momentum_window
        self.vol_lookback = vol_lookback
        self.vol_pctl = vol_percentile
        self.bull_thresh = bull_threshold
        self.bear_thresh = bear_threshold
        self.hysteresis_days = hysteresis_days

        # State
        self._current_regime: str = "neutral"
        self._transition_count: int = 0
        self._days_in_regime: int = 0
        self._prev_scale: float = 1.0

    def detect(self, market_data: pd.DataFrame) -> RegimeState:
        """Detect regime with hysteresis and adaptive window."""
        if len(market_data) < self.vol_lookback:
            return RegimeState("neutral", 0.0, 0.0, 1.0)

        close = market_data["close"].values

        # Adaptive momentum window
        returns = np.diff(np.log(close[-self.vol_lookback:]))
        current_vol = returns[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.2
        window = self._adaptive_window(current_vol, returns)

        momentum = close[-1] / close[-window] - 1

        # High vol detection
        high_vol = self._is_high_vol(returns)

        # Mean-reversion detection
        in_reversion = self._detect_reversion(close)

        # Raw regime classification
        if momentum < self.bear_thresh:
            raw_regime = "bear"
        elif in_reversion:
            raw_regime = "mean_reversion"
        elif momentum > self.bull_thresh and not high_vol:
            raw_regime = "bull"
        elif high_vol:
            raw_regime = "volatile"
        else:
            raw_regime = "neutral"

        # Apply hysteresis
        confirmed_regime, confidence = self._apply_hysteresis(raw_regime)
        scale = self._cost_aware_scale(confirmed_regime, confidence)

        return RegimeState(
            regime=confirmed_regime,
            momentum=float(momentum),
            volatility=float(current_vol),
            scale_factor=scale,
            confidence=confidence,
            days_in_regime=self._days_in_regime,
        )

    def _apply_hysteresis(self, raw_regime: str) -> tuple[str, float]:
        """Require N consecutive days in new regime before switching."""
        if raw_regime != self._current_regime:
            self._transition_count += 1

            if self._transition_count >= self.hysteresis_days:
                # Confirmed transition
                self._current_regime = raw_regime
                self._transition_count = 0
                self._days_in_regime = 1
                return raw_regime, 1.0
            else:
                # Not confirmed yet — stay in current regime
                confidence = 1.0 - (self._transition_count / self.hysteresis_days) * 0.3
                return self._current_regime, confidence
        else:
            self._transition_count = 0
            self._days_in_regime += 1
            return self._current_regime, 1.0

    def _cost_aware_scale(self, regime: str, confidence: float) -> float:
        """Don't liquidate for low-confidence bear calls (cost > benefit)."""
        base_scale = self.SCALE_FACTORS.get(regime, 1.0)

        # If going to bear (cash), require high confidence
        if regime == "bear" and base_scale == 0.0:
            if confidence < 0.80:
                return 0.5   # Partial reduction, not full liquidation

        # Smooth transition: don't jump from 1.2 to 0.6 in one day
        if abs(self._prev_scale - base_scale) > 0.3:
            # Large regime change: step halfway first
            smoothed = (self._prev_scale + base_scale) / 2
            self._prev_scale = smoothed
            return smoothed

        self._prev_scale = base_scale
        return base_scale

    def _adaptive_window(self, current_vol: float, returns: np.ndarray) -> int:
        """Shorter window in low vol (catches trends), longer in high vol (filters noise)."""
        avg_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2

        if avg_vol < 1e-8:
            return self.default_window

        vol_ratio = current_vol / avg_vol

        if vol_ratio > 1.3:    # High vol: use longer window
            return min(30, self.default_window + 10)
        elif vol_ratio < 0.7:  # Low vol: use shorter window
            return max(10, self.default_window - 10)
        return self.default_window

    def _is_high_vol(self, returns: np.ndarray) -> bool:
        """Is current vol above Nth percentile of historical?"""
        if len(returns) < 25:
            return False

        window = 20
        rolling_vols = [
            returns[i:i+window].std() * np.sqrt(252)
            for i in range(len(returns) - window)
        ]

        if not rolling_vols:
            return False

        current_vol = returns[-window:].std() * np.sqrt(252)
        pctl_rank = sum(1 for v in rolling_vols if v < current_vol) / len(rolling_vols)

        return pctl_rank >= self.vol_pctl

    def _detect_reversion(self, close: np.ndarray) -> bool:
        """Detect mean-reversion setup: price far from MA and decelerating."""
        if len(close) < 30:
            return False

        ma_20 = close[-20:].mean()
        deviation = (close[-1] - ma_20) / ma_20

        if abs(deviation) < 0.03:
            return False

        # Is deviation contracting? (mean reversion in progress)
        prev_ma = close[-21:-1].mean()
        prev_deviation = (close[-2] - prev_ma) / prev_ma
        contracting = abs(deviation) < abs(prev_deviation)

        return contracting and abs(deviation) > 0.03
