"""Sector-based portfolio optimization and allocation."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Optimizes sector allocations with risk constraints.

    Supports multiple optimization objectives:
    - Maximum Sharpe ratio
    - Minimum variance
    - Risk parity
    - Signal-weighted allocation
    """

    def __init__(
        self,
        n_sectors: int = 11,
        max_position_pct: float = 0.10,
        risk_free_rate: float = 0.03,
    ):
        self.n_sectors = n_sectors
        self.max_position = max_position_pct
        self.risk_free_rate = risk_free_rate / 252  # Daily

    def optimize(
        self,
        signals: np.ndarray,
        sector_returns: pd.DataFrame,
        method: str = "signal_weighted",
    ) -> np.ndarray:
        """Compute optimal sector weights.

        Args:
            signals: (n_sectors,) trading signals [-1, 1]
            sector_returns: Historical sector returns
            method: Optimization method

        Returns:
            (n_sectors,) allocation weights
        """
        if method == "signal_weighted":
            weights = self._signal_weighted(signals)
        elif method == "max_sharpe":
            weights = self._max_sharpe(sector_returns)
        elif method == "min_variance":
            weights = self._min_variance(sector_returns)
        elif method == "risk_parity":
            weights = self._risk_parity(sector_returns)
        else:
            weights = self._signal_weighted(signals)

        # Apply position limits
        weights = np.clip(weights, -self.max_position, self.max_position)

        return weights

    def _signal_weighted(self, signals: np.ndarray) -> np.ndarray:
        """Allocate proportionally to signal strength."""
        abs_sum = np.abs(signals).sum()
        if abs_sum > 0:
            return signals / abs_sum * min(abs_sum, 1.0)
        return np.zeros(self.n_sectors)

    def _max_sharpe(self, returns: pd.DataFrame) -> np.ndarray:
        """Maximize Sharpe ratio using scipy optimization."""
        mean_ret = returns.mean().values
        cov = returns.cov().values
        n = self.n_sectors

        def neg_sharpe(w):
            port_ret = np.dot(w, mean_ret) - self.risk_free_rate
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w))) + 1e-8
            return -port_ret / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]
        bounds = [(-self.max_position, self.max_position)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        return result.x if result.success else x0

    def _min_variance(self, returns: pd.DataFrame) -> np.ndarray:
        """Minimize portfolio variance."""
        cov = returns.cov().values
        n = self.n_sectors

        def portfolio_var(w):
            return np.dot(w, np.dot(cov, w))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, self.max_position)] * n
        x0 = np.ones(n) / n

        result = minimize(
            portfolio_var, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        return result.x if result.success else x0

    def _risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Equal risk contribution across sectors."""
        cov = returns.cov().values
        n = self.n_sectors

        def risk_budget_obj(w):
            port_var = np.dot(w, np.dot(cov, w))
            marginal_contrib = np.dot(cov, w)
            risk_contrib = w * marginal_contrib
            target = port_var / n
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, self.max_position)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_budget_obj, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        return result.x if result.success else x0

    def compute_portfolio_metrics(
        self,
        weights: np.ndarray,
        sector_returns: pd.DataFrame,
    ) -> dict:
        """Compute expected portfolio metrics for given weights."""
        mean_ret = sector_returns.mean().values
        cov = sector_returns.cov().values

        port_return = np.dot(weights, mean_ret) * 252
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights))) * np.sqrt(252)
        sharpe = (port_return - self.risk_free_rate * 252) / (port_vol + 1e-8)

        return {
            "expected_annual_return": port_return,
            "expected_annual_volatility": port_vol,
            "expected_sharpe": sharpe,
            "gross_exposure": np.abs(weights).sum(),
            "net_exposure": weights.sum(),
            "n_long": (weights > 0.001).sum(),
            "n_short": (weights < -0.001).sum(),
        }
