"""Gymnasium trading environment for sector-based portfolio management.

The agent observes discovered indicators and decides sector allocations.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class SectorTradingEnv(gym.Env):
    """Multi-sector trading environment.

    State: [discovered_indicators, portfolio_state, market_context]
    Action: sector allocation weights in [-1, 1] for each sector
    Reward: risk-adjusted returns (Sharpe/Sortino)

    The environment simulates realistic trading with:
    - Transaction costs
    - Slippage
    - Position limits
    - Drawdown tracking
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        sector_returns: np.ndarray,
        n_sectors: int = 11,
        initial_capital: float = 1e8,
        transaction_cost: float = 0.002,
        max_position: float = 1.0,
        reward_type: str = "sharpe",
        risk_penalty: float = 1.0,
        drawdown_penalty: float = 5.0,
        window_size: int = 60,
        vol_target: float = 0.15,
        episode_length: int = 252,
    ):
        """
        Args:
            features: (n_steps, feature_dim) state features (from models)
            sector_returns: (n_steps, n_sectors) daily sector returns
            n_sectors: Number of GICS sectors
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            max_position: Max absolute position per sector
            reward_type: "sharpe", "sortino", or "returns"
            risk_penalty: Penalty weight for excess volatility
            drawdown_penalty: Penalty weight for drawdown (non-linear)
            window_size: Window for rolling Sharpe computation
            vol_target: Annual volatility target (excess vol is penalized)
        """
        super().__init__()

        self.features = features.astype(np.float32)
        self.sector_returns = sector_returns.astype(np.float32)
        self.n_sectors = n_sectors
        self.n_steps = len(features)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_type = reward_type
        self.risk_penalty = risk_penalty
        self.drawdown_penalty = drawdown_penalty
        self.window_size = window_size
        self.vol_target = vol_target
        # Episode length: how many steps per episode (random window sampling)
        self.episode_length = min(episode_length, len(features))

        feature_dim = features.shape[1]
        # State: features + current positions + portfolio value ratio + step ratio
        state_dim = feature_dim + n_sectors + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        # Action: allocation per sector in [-1, 1]
        self.action_space = spaces.Box(
            low=-max_position,
            high=max_position,
            shape=(n_sectors,),
            dtype=np.float32,
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Random start: sample a different market window each episode
        max_start = max(0, self.n_steps - self.episode_length)
        self.episode_start = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
        self.episode_end = self.episode_start + self.episode_length
        self.current_step = self.episode_start

        self.positions = np.zeros(self.n_sectors, dtype=np.float32)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.returns_history = []
        self.trade_history = []

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step.

        Args:
            action: (n_sectors,) target sector allocations

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, -self.max_position, self.max_position)

        # Project action onto simplex: long-only, weights sum to 1 (no leverage)
        action = np.clip(action, 0, None)
        action_sum = action.sum()
        if action_sum > 1e-8:
            action = action / action_sum
        else:
            action = np.ones(self.n_sectors, dtype=np.float32) / self.n_sectors

        # Compute trading costs
        position_change = np.abs(action - self.positions)
        costs = np.sum(position_change) * self.transaction_cost * self.portfolio_value

        # Compute portfolio return using current episode step
        if self.current_step < self.n_steps:
            sector_ret = self.sector_returns[self.current_step]
            portfolio_return = np.dot(action, sector_ret)
            pnl = portfolio_return * self.portfolio_value - costs
        else:
            pnl = -costs

        # Update state
        self.portfolio_value += pnl
        daily_return = pnl / max(self.portfolio_value - pnl, 1.0)
        self.returns_history.append(daily_return)

        # Update peak and drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_dd = (self.peak_value - self.portfolio_value) / self.peak_value

        # Update positions
        self.positions = action.copy()
        self.current_step += 1

        # Record trade
        self.trade_history.append({
            "step": self.current_step,
            "positions": action.copy(),
            "portfolio_value": self.portfolio_value,
            "daily_return": daily_return,
            "drawdown": current_dd,
            "costs": costs,
        })

        # Compute reward
        reward = self._compute_reward(daily_return, current_dd)

        # Termination: end of episode window or bankruptcy
        terminated = self.portfolio_value <= 0
        truncated = self.current_step >= self.episode_end

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        step_idx = min(self.current_step, self.n_steps - 1)
        features = self.features[step_idx]
        value_ratio = np.float32(self.portfolio_value / self.initial_capital)
        # Step ratio relative to episode window (not total dataset)
        episode_steps = self.episode_end - self.episode_start
        step_ratio = np.float32((self.current_step - self.episode_start) / max(episode_steps, 1))

        return np.concatenate([
            features,
            self.positions,
            [value_ratio, step_ratio],
        ])

    def _compute_reward(
        self,
        daily_return: float,
        drawdown: float,
    ) -> float:
        """Compute risk-adjusted reward."""
        if self.reward_type == "sharpe":
            if len(self.returns_history) >= self.window_size:
                recent = np.array(self.returns_history[-self.window_size :])
                mean_ret = np.mean(recent)
                std_ret = np.std(recent) + 1e-8
                reward = mean_ret / std_ret * np.sqrt(252)
            else:
                reward = daily_return * 100
        elif self.reward_type == "sortino":
            if len(self.returns_history) >= self.window_size:
                recent = np.array(self.returns_history[-self.window_size :])
                mean_ret = np.mean(recent)
                downside = np.std(recent[recent < 0]) + 1e-8
                reward = mean_ret / downside * np.sqrt(252)
            else:
                reward = daily_return * 100
        else:  # returns
            reward = daily_return * 100

        # Excess volatility penalty (only penalize vol above target)
        if len(self.returns_history) >= self.window_size:
            recent = np.array(self.returns_history[-self.window_size:])
            port_vol = np.std(recent) * np.sqrt(252)
            vol_excess = max(0.0, port_vol - self.vol_target)
            reward -= self.risk_penalty * vol_excess

        # Non-linear drawdown penalty (multi-tier)
        if drawdown > 0.30:
            reward -= self.drawdown_penalty * drawdown * 10  # Crisis
        elif drawdown > 0.20:
            reward -= self.drawdown_penalty * drawdown * 5   # High risk
        elif drawdown > 0.10:
            reward -= self.drawdown_penalty * drawdown * 2   # Warning
        else:
            reward -= self.drawdown_penalty * drawdown        # Normal

        # Clip reward to prevent exploding value loss
        return float(np.clip(reward, -10.0, 10.0))

    def _get_info(self) -> dict:
        return {
            "portfolio_value": self.portfolio_value,
            "total_return": (self.portfolio_value - self.initial_capital) / self.initial_capital,
            "n_trades": len(self.trade_history),
            "max_drawdown": max(
                (t["drawdown"] for t in self.trade_history), default=0.0
            ),
        }
