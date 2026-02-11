"""Trading signal generation from model outputs."""

import numpy as np
import pandas as pd
from loguru import logger


class SignalGenerator:
    """Generates actionable trading signals from model predictions.

    Combines model ensemble outputs with sector momentum to produce
    final sector allocation signals.
    """

    def __init__(
        self,
        n_sectors: int = 11,
        signal_threshold: float = 0.0,
        momentum_window: int = 20,
        smoothing_window: int = 5,
    ):
        self.n_sectors = n_sectors
        self.signal_threshold = signal_threshold
        self.momentum_window = momentum_window
        self.smoothing_window = smoothing_window
        self.signal_history: list[np.ndarray] = []

    def generate(
        self,
        model_prediction: np.ndarray,
        rl_allocation: np.ndarray,
        sector_returns: pd.DataFrame = None,
    ) -> dict:
        """Generate final trading signals.

        Args:
            model_prediction: (n_sectors,) return predictions from transformer
            rl_allocation: (n_sectors,) allocation from RL agent
            sector_returns: Historical sector returns for momentum

        Returns:
            Dict with signal details
        """
        # Normalize predictions to [-1, 1]
        if np.std(model_prediction) > 1e-8:
            pred_signal = model_prediction / (np.abs(model_prediction).max() + 1e-8)
        else:
            pred_signal = np.zeros(self.n_sectors)

        # Combine model prediction and RL allocation
        alpha = 0.6  # Weight on RL allocation
        raw_signal = alpha * rl_allocation + (1 - alpha) * pred_signal

        # Add momentum overlay if available
        if sector_returns is not None and len(sector_returns) >= self.momentum_window:
            momentum = self._compute_momentum(sector_returns)
            raw_signal = 0.7 * raw_signal + 0.3 * momentum

        # Smooth signal
        self.signal_history.append(raw_signal)
        if len(self.signal_history) > self.smoothing_window:
            self.signal_history = self.signal_history[-self.smoothing_window:]
        smoothed = np.mean(self.signal_history, axis=0)

        # Apply threshold
        final_signal = np.where(
            np.abs(smoothed) > self.signal_threshold,
            smoothed,
            0.0,
        )

        # Normalize to sum to <= 1 (long) and >= -1 (short)
        long_sum = final_signal[final_signal > 0].sum()
        short_sum = abs(final_signal[final_signal < 0].sum())
        if long_sum > 1.0:
            final_signal[final_signal > 0] /= long_sum
        if short_sum > 1.0:
            final_signal[final_signal < 0] /= short_sum

        return {
            "signal": final_signal,
            "raw_signal": raw_signal,
            "prediction_signal": pred_signal,
            "rl_signal": rl_allocation,
            "confidence": self._compute_confidence(raw_signal, final_signal),
        }

    def _compute_momentum(self, sector_returns: pd.DataFrame) -> np.ndarray:
        """Compute sector momentum signal."""
        recent = sector_returns.iloc[-self.momentum_window:]
        cumulative = (1 + recent).prod() - 1

        # Cross-sectional z-score
        mean = cumulative.mean()
        std = cumulative.std() + 1e-8
        z_scores = (cumulative - mean) / std

        return np.clip(z_scores.values / 3, -1, 1)  # Scale to [-1, 1]

    @staticmethod
    def _compute_confidence(raw: np.ndarray, final: np.ndarray) -> float:
        """Signal confidence based on signal strength and consistency."""
        strength = np.mean(np.abs(final))
        consistency = 1.0 - np.std(raw - final)
        return float(np.clip(strength * consistency, 0, 1))

    def get_sector_rotation_signal(
        self,
        sector_returns: pd.DataFrame,
        lookback: int = 60,
        top_n: int = 3,
    ) -> dict[str, float]:
        """Generate sector rotation signal.

        Identifies sectors to overweight and underweight based
        on relative momentum.
        """
        if len(sector_returns) < lookback:
            return {}

        recent = sector_returns.iloc[-lookback:]
        momentum = (1 + recent).prod() - 1
        ranked = momentum.sort_values(ascending=False)

        signals = {}
        for i, (sector, mom) in enumerate(ranked.items()):
            if i < top_n:
                signals[sector] = 1.0  # Overweight
            elif i >= len(ranked) - top_n:
                signals[sector] = -1.0  # Underweight
            else:
                signals[sector] = 0.0  # Neutral

        return signals
