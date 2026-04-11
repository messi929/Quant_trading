"""Rule-based direction engine — momentum + flow + event + mean-reversion.

Medallion-inspired enhancements:
  1. Regime-adaptive signal weighting (bull→momentum, bear→reduce, volatile→event)
  2. Mean-reversion signal for KRX overbought/oversold detection
  3. Flow persistence check (sustained vs. one-day spike)
  4. Signal agreement penalty (conflicting signals → reduced clarity)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DirectionSignal:
    direction: str       # "long", "short", "neutral"
    clarity: float       # [0, 1] — how clear the direction is
    momentum_score: float
    flow_score: float
    event_score: float
    reversion_score: float = 0.0
    signal_agreement: float = 1.0   # 1.0 = all agree, 0.33 = conflicting


class DirectionEngine:
    """Determines trade direction using rule-based signals.

    Regime-adaptive: weights shift based on market regime.
    """

    def __init__(
        self,
        momentum_window: int = 20,
        momentum_weight: float = 0.5,
        flow_weight: float = 0.3,
        event_weight: float = 0.2,
        min_direction_clarity: float = 0.3,
    ):
        self.momentum_window = momentum_window
        self.min_clarity = min_direction_clarity

        # Config weights used as neutral-regime baseline
        # Other regimes shift relative to this baseline
        base_m, base_f, base_e = momentum_weight, flow_weight, event_weight
        base_r = max(0, 1.0 - base_m - base_f - base_e)  # Remainder to reversion

        self.regime_weights = {
            "neutral":        {"momentum": base_m, "flow": base_f, "event": base_e, "reversion": base_r},
            "bull":           {"momentum": min(base_m + 0.10, 0.60), "flow": base_f - 0.05, "event": max(base_e - 0.05, 0.05), "reversion": base_r},
            "bear":           {"momentum": max(base_m - 0.25, 0.15), "flow": base_f, "event": base_e, "reversion": min(base_r + 0.25, 0.40)},
            "volatile":       {"momentum": max(base_m - 0.20, 0.15), "flow": base_f - 0.05, "event": min(base_e + 0.15, 0.35), "reversion": base_r + 0.10},
            "mean_reversion": {"momentum": max(base_m - 0.30, 0.10), "flow": base_f - 0.05, "event": base_e, "reversion": min(base_r + 0.35, 0.50)},
        }

    def evaluate(
        self,
        ticker: str,
        ticker_data: pd.DataFrame,
        event_score: float = 0.0,
        regime: str = "neutral",
    ) -> DirectionSignal:
        """Evaluate direction for a single ticker with regime-adaptive weighting."""
        momentum = self._momentum_score(ticker_data)
        flow = self._flow_score(ticker_data)
        reversion = self._mean_reversion_score(ticker_data)
        event = event_score

        # Regime-adaptive weights (derived from config baseline)
        weights = self.regime_weights.get(regime, self.regime_weights["neutral"])

        combined = (
            weights["momentum"] * momentum
            + weights["flow"] * flow
            + weights["event"] * event
            + weights["reversion"] * reversion
        )

        # Signal agreement: do components agree on direction?
        agreement = self._signal_agreement(momentum, flow, event, reversion)

        # Clarity penalizes disagreement
        base_clarity = min(abs(combined), 1.0)
        clarity = base_clarity * (0.5 + 0.5 * agreement)

        if combined > 0:
            direction = "long"
        elif combined < 0:
            direction = "short"
        else:
            direction = "neutral"

        return DirectionSignal(
            direction=direction,
            clarity=clarity,
            momentum_score=momentum,
            flow_score=flow,
            event_score=event,
            reversion_score=reversion,
            signal_agreement=agreement,
        )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        event_scores: dict[str, float] | None = None,
        regime: str = "neutral",
    ) -> dict[str, DirectionSignal]:
        """Evaluate direction for all tickers."""
        event_scores = event_scores or {}
        results = {}

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            if len(group) < self.momentum_window:
                continue

            event = event_scores.get(ticker, 0.0)
            signal = self.evaluate(ticker, group, event_score=event, regime=regime)
            results[ticker] = signal

        return results

    def _momentum_score(self, data: pd.DataFrame) -> float:
        """Vol-regime-adjusted momentum score in [-1, 1]."""
        if len(data) < self.momentum_window:
            return 0.0

        close = data["close"].values
        recent_return = (close[-1] / close[-self.momentum_window] - 1)

        returns = np.diff(np.log(close[-60:])) if len(close) >= 60 else np.diff(np.log(close))
        if len(returns) < 5:
            return 0.0

        vol = returns.std() * np.sqrt(252)
        if vol < 1e-8:
            return 0.0

        z = recent_return / (vol / np.sqrt(252) * np.sqrt(self.momentum_window))

        # Vol-regime adjustment: attenuate in high-vol (false breakouts)
        if vol > 0.35:
            z *= 0.7
        elif vol < 0.10:
            z *= 1.3

        return float(np.clip(np.tanh(z / 2), -1, 1))

    def _flow_score(self, data: pd.DataFrame) -> float:
        """Flow score with persistence check.

        Sustained flow (3+ consecutive days same sign) = stronger signal.
        One-day spike = discounted.
        """
        if "foreign_net_ratio" not in data.columns:
            return 0.0

        recent = data.tail(10)
        if len(recent) < 3:
            return 0.0

        foreign_vals = recent["foreign_net_ratio"].values
        inst_col = "inst_net_ratio"
        inst_vals = recent[inst_col].values if inst_col in recent.columns else np.zeros(len(recent))

        # Weighted mean: recent days matter more
        weights = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.20])
        weights = weights[-len(foreign_vals):]
        weights /= weights.sum()

        foreign_avg = np.average(foreign_vals, weights=weights)
        inst_avg = np.average(inst_vals, weights=weights)

        # Persistence: are last 3 days same sign?
        last3_foreign = foreign_vals[-3:]
        persistence = 1.0
        if (last3_foreign > 0).all() or (last3_foreign < 0).all():
            persistence = 1.3   # Sustained flow → amplify
        elif np.sign(last3_foreign[-1]) != np.sign(last3_foreign[-2]):
            persistence = 0.6   # Reversal → discount

        combined = (foreign_avg * 0.6 + inst_avg * 0.4) * persistence
        return float(np.clip(np.tanh(combined * 2), -1, 1))

    def _mean_reversion_score(self, data: pd.DataFrame) -> float:
        """Mean-reversion signal for KRX stocks.

        Korean stocks show strong mean-reversion when price deviates >2σ from MA.
        Returns: negative = overbought (expect down), positive = oversold (expect up).
        """
        if len(data) < 30:
            return 0.0

        close = data["close"].values
        ma_20 = close[-20:].mean()
        deviation = (close[-1] - ma_20) / ma_20

        returns = np.diff(np.log(close[-30:]))
        sigma = returns.std()
        if sigma < 1e-8:
            return 0.0

        zscore = deviation / sigma

        # Only trigger on extremes (|z| > 2.0)
        if abs(zscore) < 2.0:
            return 0.0

        # Contrarian: overbought → short bias, oversold → long bias
        # But cap at ±0.5 (supplement, not dominant)
        reversion = float(np.clip(np.tanh(-zscore / 3), -0.5, 0.5))

        # Verify momentum is actually decelerating (don't fight trends)
        if len(close) >= 5:
            recent_accel = (close[-1] / close[-3]) - (close[-3] / close[-5])
            if abs(zscore) > 2.5 and np.sign(recent_accel) == np.sign(deviation):
                # Momentum still accelerating → don't fade it yet
                reversion *= 0.3

        return reversion

    @staticmethod
    def _signal_agreement(momentum: float, flow: float, event: float, reversion: float) -> float:
        """Measure agreement between signals. 1.0 = all agree, ~0.3 = conflicting.

        When momentum says long but flow says short, clarity should drop.
        """
        threshold = 0.1
        signals = []
        for s in [momentum, flow, event, reversion]:
            if s > threshold:
                signals.append(1)
            elif s < -threshold:
                signals.append(-1)
            else:
                signals.append(0)

        # Remove neutral signals from agreement calc
        active = [s for s in signals if s != 0]
        if len(active) <= 1:
            return 0.7   # Too few active signals → moderate clarity

        # Count majority direction
        pos = sum(1 for s in active if s > 0)
        neg = sum(1 for s in active if s < 0)
        majority = max(pos, neg)

        return majority / len(active)
