"""Risk management: position sizing, stop-loss, drawdown control."""

import numpy as np
import pandas as pd
from loguru import logger


class RiskManager:
    """Manages portfolio risk with position limits and stop-losses.

    Implements multiple risk controls:
    - Maximum position size per sector
    - Portfolio-level drawdown limit
    - Volatility-based position sizing
    - Correlation-based exposure limits
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_drawdown: float = 0.15,
        max_daily_loss: float = 0.03,
        max_portfolio_vol: float = 0.20,
        vol_lookback: int = 20,
    ):
        self.max_position = max_position_pct
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_vol = max_portfolio_vol / np.sqrt(252)  # Daily target
        self.vol_lookback = vol_lookback

        self.peak_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.is_risk_off: bool = False

    def check_and_adjust(
        self,
        weights: np.ndarray,
        portfolio_value: float,
        sector_returns: pd.DataFrame = None,
    ) -> tuple[np.ndarray, dict]:
        """Check risk limits and adjust positions if needed.

        Args:
            weights: (n_sectors,) proposed allocation weights
            portfolio_value: Current portfolio value
            sector_returns: Recent sector returns for vol estimation

        Returns:
            Adjusted weights and risk report dict
        """
        report = {"adjustments": [], "risk_level": "normal"}

        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)
        current_dd = (self.peak_value - portfolio_value) / self.peak_value

        # Check drawdown limit
        if current_dd > self.max_drawdown:
            scale = 0.5  # Cut positions in half
            weights = weights * scale
            report["adjustments"].append(
                f"Drawdown limit ({current_dd:.1%} > {self.max_drawdown:.1%}): "
                f"positions scaled to {scale:.0%}"
            )
            report["risk_level"] = "high"
            self.is_risk_off = True

        # Gradual recovery from risk-off
        if self.is_risk_off and current_dd < self.max_drawdown * 0.5:
            self.is_risk_off = False
            report["adjustments"].append("Exiting risk-off mode")

        # Position limits
        clipped = np.clip(weights, -self.max_position, self.max_position)
        if not np.allclose(weights, clipped):
            report["adjustments"].append("Position limits applied")
        weights = clipped

        # Volatility targeting
        if sector_returns is not None and len(sector_returns) >= self.vol_lookback:
            weights, vol_adj = self._vol_target(weights, sector_returns)
            if vol_adj != 1.0:
                report["adjustments"].append(
                    f"Vol targeting: scale={vol_adj:.2f}"
                )

        report["current_drawdown"] = current_dd
        report["gross_exposure"] = np.abs(weights).sum()
        report["net_exposure"] = weights.sum()
        report["is_risk_off"] = self.is_risk_off

        return weights, report

    def _vol_target(
        self,
        weights: np.ndarray,
        sector_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, float]:
        """Scale positions to target portfolio volatility."""
        recent = sector_returns.iloc[-self.vol_lookback:]
        cov = recent.cov().values

        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

        if port_vol > self.max_portfolio_vol and port_vol > 1e-8:
            scale = self.max_portfolio_vol / port_vol
            return weights * scale, scale
        return weights, 1.0

    def compute_var(
        self,
        weights: np.ndarray,
        sector_returns: pd.DataFrame,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """Compute Value at Risk.

        Args:
            weights: Portfolio weights
            sector_returns: Historical returns
            confidence: VaR confidence level
            method: "historical" or "parametric"

        Returns:
            VaR as positive number (max expected loss)
        """
        portfolio_returns = sector_returns.values @ weights

        if method == "historical":
            var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        else:  # parametric
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            from scipy.stats import norm
            z = norm.ppf(confidence)
            var = -(mean - z * std)

        return max(var, 0.0)

    def compute_expected_shortfall(
        self,
        weights: np.ndarray,
        sector_returns: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """Compute Expected Shortfall (CVaR)."""
        portfolio_returns = sector_returns.values @ weights
        var_threshold = np.percentile(
            portfolio_returns, (1 - confidence) * 100
        )
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        if len(tail_returns) > 0:
            return -tail_returns.mean()
        return 0.0

    def position_sizing(
        self,
        signal_strength: float,
        volatility: float,
        base_size: float = 0.05,
    ) -> float:
        """Kelly-inspired position sizing based on signal and vol.

        Args:
            signal_strength: Signal confidence [0, 1]
            volatility: Asset/sector volatility
            base_size: Base position size

        Returns:
            Recommended position size
        """
        if volatility < 1e-8:
            return 0.0

        # Half-Kelly for conservatism
        kelly = signal_strength / (volatility * 100)
        size = base_size * min(kelly * 0.5, 2.0)  # Cap at 2x base

        return np.clip(size, 0, self.max_position)
