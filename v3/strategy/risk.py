"""Portfolio risk management — adaptive circuit breaker, tail risk, vol-of-vol.

Medallion-inspired enhancements:
  1. Adaptive circuit breaker (model IC quality + recovery progress)
  2. Vol-of-vol shock detector (VIX-like sudden expansion)
  3. 99th percentile tail risk (VaR) estimation
  4. Portfolio-deployed-proportional daily limits
  5. Stress test scenarios
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class RiskManager:
    """Portfolio-level risk management with adaptive controls."""

    def __init__(
        self,
        daily_loss_limit: float = 0.015,
        total_loss_limit: float = 0.15,
        circuit_breaker_levels: dict | None = None,
    ):
        self.daily_limit = daily_loss_limit
        self.total_limit = total_loss_limit
        self.cb_levels = circuit_breaker_levels or {
            "caution": 0.05,
            "warning": 0.10,
            "high": 0.20,
            "crisis": 0.30,
        }

        self.peak_value: float = 0.0
        self.prev_mdd: float = 0.0
        self.vol_history: list[float] = []

    def update(self, portfolio_value: float) -> None:
        self.peak_value = max(self.peak_value, portfolio_value)

    def mdd(self, portfolio_value: float) -> float:
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - portfolio_value) / self.peak_value

    # ── Circuit Breaker ──────────────────────────────────────

    def circuit_breaker_scale(
        self,
        portfolio_value: float,
        model_ic: float = 0.30,
        position_concentration: float = 0.0,
    ) -> float:
        """Adaptive circuit breaker: accounts for model quality and recovery.

        Returns position scale [0, 1].
        """
        if self.peak_value <= 0:
            return 1.0

        current_mdd = self.mdd(portfolio_value)

        # Base scale from MDD levels
        base_scale = self._base_cb_scale(current_mdd)

        # Adjust for model quality
        if model_ic < 0.25:
            base_scale *= 0.7
        elif model_ic > 0.40:
            base_scale = min(1.0, base_scale * 1.1)

        # Adjust for concentration
        if position_concentration > 0.50:
            base_scale *= 0.8

        # Recovery credit: if MDD is contracting, relax slightly
        if self.prev_mdd > 0 and current_mdd < self.prev_mdd * 0.8:
            recovery_bonus = 0.1
            base_scale = min(1.0, base_scale + recovery_bonus)

        self.prev_mdd = current_mdd
        return max(0.0, min(1.0, base_scale))

    def _base_cb_scale(self, mdd: float) -> float:
        if mdd >= self.cb_levels["crisis"]:
            return 0.0
        if mdd >= self.cb_levels["high"]:
            return 0.25
        if mdd >= self.cb_levels["warning"]:
            return 0.50
        if mdd >= self.cb_levels["caution"]:
            return 0.75
        return 1.0

    # ── Daily/Total Limits ───────────────────────────────────

    def check_daily_limit(self, daily_return: float, portfolio_deployed: float = 0.5) -> bool:
        """Proportional daily limit: tighter when more deployed."""
        limit = self._dynamic_daily_limit(portfolio_deployed)
        if daily_return <= -limit:
            logger.warning(f"Daily loss limit breached: {daily_return:.2%} (limit={limit:.2%})")
            return True
        return False

    def _dynamic_daily_limit(self, deployed: float) -> float:
        """Tighter limits when more capital is at risk."""
        if deployed > 0.90:
            return 0.010
        elif deployed > 0.70:
            return 0.012
        elif deployed < 0.30:
            return 0.020
        return self.daily_limit

    def check_total_limit(self, total_return: float) -> bool:
        if total_return <= -self.total_limit:
            logger.warning(f"Total loss limit breached: {total_return:.2%}")
            return True
        return False

    # ── Vol-of-Vol Shock ─────────────────────────────────────

    def check_vol_shock(self, current_vol: float) -> float:
        """Scale positions down during vol-of-vol shocks.

        Returns scale factor [0.1, 1.0].
        """
        self.vol_history.append(current_vol)
        if len(self.vol_history) > 252:
            self.vol_history = self.vol_history[-252:]

        if len(self.vol_history) < 20:
            return 1.0

        hist = np.array(self.vol_history[:-1])
        hist_mean = hist.mean()
        hist_std = hist.std()

        if hist_std < 1e-8:
            return 1.0

        vol_zscore = (current_vol - hist_mean) / hist_std

        if vol_zscore > 3.0:
            logger.warning(f"VOL SHOCK: z={vol_zscore:.1f} → scale 0.3")
            return 0.3
        elif vol_zscore > 2.0:
            logger.info(f"Vol elevated: z={vol_zscore:.1f} → scale 0.5")
            return 0.5
        elif vol_zscore > 1.5:
            return 0.7

        return 1.0

    # ── Tail Risk (VaR) ──────────────────────────────────────

    def estimate_tail_risk(
        self,
        position_weights: list[float],
        position_vols: list[float],
        correlations: np.ndarray | None = None,
        stress_vol_mult: float = 1.5,
        stress_corr_shift: float = 0.5,
    ) -> float:
        """Estimate 99th percentile portfolio loss (1-day VaR).

        Uses stressed vol and correlation for conservative estimate.

        Returns:
            Estimated worst 1-day loss as positive fraction (e.g., 0.03 = 3% loss).
        """
        n = len(position_weights)
        if n == 0:
            return 0.0

        weights = np.array(position_weights)
        vols = np.array(position_vols)

        if correlations is None:
            correlations = np.full((n, n), 0.3)
            np.fill_diagonal(correlations, 1.0)

        # Stress: increase vols and correlations
        stressed_vols = vols * stress_vol_mult
        stressed_corr = (1 - stress_corr_shift) * correlations + stress_corr_shift * np.ones((n, n))
        np.fill_diagonal(stressed_corr, 1.0)

        # Daily vol
        daily_vols = stressed_vols / np.sqrt(252)
        cov = np.diag(daily_vols) @ stressed_corr @ np.diag(daily_vols)
        port_vol = np.sqrt(weights @ cov @ weights)

        # 99th percentile (z=2.33)
        var_99 = 2.33 * port_vol

        return float(var_99)

    def check_tail_risk_limit(
        self,
        var_99: float,
        max_var: float = 0.05,
    ) -> bool:
        """Returns True if VaR exceeds limit (should reduce positions)."""
        if var_99 > max_var:
            logger.warning(f"Tail risk exceeded: VaR99={var_99:.2%} > limit={max_var:.2%}")
            return True
        return False
