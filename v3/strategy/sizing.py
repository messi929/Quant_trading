"""Volatility-targeting position sizing with correlation and Kelly.

Medallion-inspired enhancements:
  1. Covariance-aware portfolio sizing (joint vol targeting)
  2. Half-Kelly non-linear scaling based on empirical edge
  3. Liquidity-constrained sizing (volume participation limit)
  4. ATR adjustment for current market conditions
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class VolTargetSizer:
    """Position sizing with portfolio-level vol targeting and Kelly."""

    def __init__(
        self,
        target_annual_vol: float = 0.15,
        max_single_weight: float = 0.40,
        min_position_weight: float = 0.15,
        volume_participation_max: float = 0.05,
        kelly_fraction: float = 0.5,  # Half-Kelly
    ):
        self.target_vol = target_annual_vol
        self.max_weight = max_single_weight
        self.min_weight = min_position_weight
        self.max_participation = volume_participation_max
        self.kelly_frac = kelly_fraction

    def size_single(
        self,
        predicted_vol: float,
        confidence: float = 0.5,
        n_positions: int = 1,
    ) -> float:
        """Size a single position using inverse-vol + Kelly."""
        if predicted_vol <= 0.01:
            predicted_vol = 0.20

        # Base: inverse vol sizing
        base = self.target_vol / predicted_vol

        # Kelly scaling (non-linear): edge = (confidence - 0.5) * 2
        edge = max((confidence - 0.5) * 2, 0)
        kelly_scale = self.kelly_frac * edge   # Half-Kelly on the edge
        # Combine: base × (0.3 + 0.7 * kelly_scale)
        scaled = base * (0.3 + 0.7 * min(kelly_scale, 1.0))

        # Equal-weight constraint
        equal = 0.95 / max(n_positions, 1)
        weight = min(scaled, equal, self.max_weight)
        weight = max(weight, self.min_weight)

        return round(weight, 4)

    def size_portfolio(
        self,
        predicted_vols: dict[str, float],
        confidences: dict[str, float],
        correlations: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Size all positions jointly, accounting for correlation.

        Args:
            predicted_vols: {ticker: annualized vol}.
            confidences: {ticker: confidence [0,1]}.
            correlations: (n, n) correlation matrix (or None for independent).

        Returns:
            {ticker: weight}.
        """
        tickers = list(predicted_vols.keys())
        n = len(tickers)

        if n == 0:
            return {}
        if n == 1:
            t = tickers[0]
            return {t: self.size_single(predicted_vols[t], confidences[t], 1)}

        vols = np.array([max(predicted_vols[t], 0.01) for t in tickers])
        confs = np.array([confidences[t] for t in tickers])

        # Default correlation: moderate positive (same market)
        if correlations is None:
            correlations = np.full((n, n), 0.3)
            np.fill_diagonal(correlations, 1.0)

        # Inverse-vol weights
        inv_vols = 1.0 / vols
        raw_weights = inv_vols / inv_vols.sum()

        # Compute portfolio vol with these weights
        cov_matrix = np.diag(vols / np.sqrt(252)) @ correlations @ np.diag(vols / np.sqrt(252))
        port_vol = np.sqrt(raw_weights @ cov_matrix @ raw_weights) * np.sqrt(252)

        # Scale to target vol
        if port_vol > 0:
            scale = self.target_vol / port_vol
        else:
            scale = 1.0
        adjusted = raw_weights * scale

        # Kelly adjustment per ticker
        for i, t in enumerate(tickers):
            edge = max((confs[i] - 0.5) * 2, 0)
            kelly_adj = 0.3 + 0.7 * min(self.kelly_frac * edge, 1.0)
            adjusted[i] *= kelly_adj

        # Correlation drag: reduce co-moving positions
        for i in range(n):
            avg_corr = (correlations[i, :].sum() - 1.0) / max(n - 1, 1)
            drag = max(0.5, 1.0 - avg_corr * 0.3)
            adjusted[i] *= drag

        # Apply bounds
        adjusted = np.clip(adjusted, self.min_weight, self.max_weight)

        # Normalize to 95%
        total = adjusted.sum()
        if total > 0.95:
            adjusted = adjusted / total * 0.95

        return {t: round(float(adjusted[i]), 4) for i, t in enumerate(tickers)}

    def apply_liquidity_constraint(
        self,
        weight: float,
        capital: float,
        stock_price: float,
        daily_volume: int,
    ) -> float:
        """Reduce sizing if position would exceed volume participation limit."""
        if daily_volume <= 0 or stock_price <= 0:
            return weight

        position_value = capital * weight
        position_shares = position_value / stock_price
        max_shares = daily_volume * self.max_participation

        if position_shares > max_shares:
            max_value = max_shares * stock_price
            constrained_weight = max_value / capital
            if constrained_weight < self.min_weight:
                return 0.0  # Skip: can't even take minimum position
            return round(min(weight, constrained_weight), 4)

        return weight

    @staticmethod
    def estimate_correlation(
        ticker_returns: dict[str, np.ndarray],
        lookback: int = 60,
    ) -> np.ndarray:
        """Estimate correlation matrix from recent returns."""
        tickers = list(ticker_returns.keys())
        n = len(tickers)

        if n <= 1:
            return np.ones((n, n))

        # Build returns matrix
        min_len = min(len(ticker_returns[t]) for t in tickers)
        min_len = min(min_len, lookback)

        returns_matrix = np.column_stack([
            ticker_returns[t][-min_len:] for t in tickers
        ])

        # Compute correlation with shrinkage toward identity
        raw_corr = np.corrcoef(returns_matrix.T)
        raw_corr = np.nan_to_num(raw_corr, nan=0.3)

        # Ledoit-Wolf-style shrinkage (simplified)
        shrinkage = 0.3  # Shrink 30% toward identity
        shrunk = (1 - shrinkage) * raw_corr + shrinkage * np.eye(n)

        return shrunk
