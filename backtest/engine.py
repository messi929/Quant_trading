"""Backtesting engine with realistic simulation of trading costs."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.signal import SignalGenerator
from strategy.portfolio import PortfolioOptimizer
from strategy.risk import RiskManager
from backtest.metrics import compute_metrics


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates realistic trading with:
    - Commission fees
    - Slippage
    - Sector rebalancing
    - Risk management integration
    """

    def __init__(
        self,
        initial_capital: float = 1e8,
        commission_rate: float = 0.00015,
        slippage_rate: float = 0.001,
        rebalance_frequency: str = "weekly",
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.rebalance_frequency = rebalance_frequency

        self.signal_gen = SignalGenerator()
        # Allow up to 50% per sector so that top-3 signals (each ~33%) are not clipped
        self.portfolio_opt = PortfolioOptimizer(max_position_pct=0.50)
        self.risk_mgr = RiskManager(max_position_pct=0.50)

    def run(
        self,
        sector_returns: pd.DataFrame,
        signals: Optional[np.ndarray] = None,
        model_predictions: Optional[np.ndarray] = None,
        rl_allocations: Optional[np.ndarray] = None,
    ) -> dict:
        """Run backtest over the given period.

        Args:
            sector_returns: (n_days, n_sectors) DataFrame of daily returns
            signals: Pre-computed signals (n_days, n_sectors)
            model_predictions: Model return predictions (n_days, n_sectors)
            rl_allocations: RL agent allocations (n_days, n_sectors)

        Returns:
            Dict with equity curve, trade log, and performance metrics
        """
        n_days = len(sector_returns)
        n_sectors = sector_returns.shape[1]
        dates = sector_returns.index

        logger.info(
            f"Running backtest: {n_days} days, {n_sectors} sectors, "
            f"capital={self.initial_capital:,.0f}"
        )

        # State tracking
        portfolio_value = self.initial_capital
        positions = np.zeros(n_sectors)
        equity_curve = [portfolio_value]
        trade_log = []
        daily_returns = []
        rebalance_days = self._get_rebalance_days(dates)

        for day in range(n_days):
            date = dates[day]
            day_return = sector_returns.iloc[day].values

            # Portfolio return (before rebalance)
            port_ret = np.dot(positions, day_return)
            pnl = port_ret * portfolio_value

            # Update portfolio value
            portfolio_value += pnl
            daily_returns.append(port_ret)

            # Rebalance check
            if date in rebalance_days:
                # Get target allocation
                if signals is not None:
                    target_signal = signals[day]
                elif model_predictions is not None and rl_allocations is not None:
                    sig = self.signal_gen.generate(
                        model_predictions[day],
                        rl_allocations[day],
                        sector_returns.iloc[max(0, day - 60) : day],
                    )
                    target_signal = sig["signal"]
                else:
                    continue

                # Portfolio optimization
                lookback = sector_returns.iloc[max(0, day - 60) : day]
                target_weights = self.portfolio_opt.optimize(
                    target_signal, lookback, method="signal_weighted"
                )

                # Risk management
                target_weights, risk_report = self.risk_mgr.check_and_adjust(
                    target_weights, portfolio_value, lookback
                )

                # Trading costs
                turnover = np.abs(target_weights - positions)
                total_turnover = turnover.sum()
                commission = total_turnover * self.commission_rate * portfolio_value
                slippage = total_turnover * self.slippage_rate * portfolio_value
                total_cost = commission + slippage

                portfolio_value -= total_cost

                # Log trade
                trade_log.append({
                    "date": date,
                    "old_positions": positions.copy(),
                    "new_positions": target_weights.copy(),
                    "turnover": total_turnover,
                    "commission": commission,
                    "slippage": slippage,
                    "total_cost": total_cost,
                    "portfolio_value": portfolio_value,
                    "risk_report": risk_report,
                })

                positions = target_weights.copy()

            equity_curve.append(portfolio_value)

        # Compute metrics
        equity_series = pd.Series(
            equity_curve[1:],  # Skip initial value
            index=dates,
        )
        returns_series = pd.Series(daily_returns, index=dates)
        metrics = compute_metrics(returns_series, equity_series, self.initial_capital)

        logger.info(
            f"Backtest complete | "
            f"Return: {metrics['total_return']:.2%} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"MDD: {metrics['max_drawdown']:.2%} | "
            f"Trades: {len(trade_log)}"
        )

        return {
            "equity_curve": equity_series,
            "daily_returns": returns_series,
            "trade_log": trade_log,
            "metrics": metrics,
            "final_positions": positions,
        }

    def _get_rebalance_days(self, dates: pd.DatetimeIndex) -> set:
        """Determine rebalance dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return set(dates)
        elif self.rebalance_frequency == "weekly":
            # Rebalance on Mondays (or first trading day of week)
            rebalance = set()
            current_week = -1
            for date in dates:
                week = date.isocalendar()[1]
                if week != current_week:
                    rebalance.add(date)
                    current_week = week
            return rebalance
        elif self.rebalance_frequency == "monthly":
            rebalance = set()
            current_month = -1
            for date in dates:
                if date.month != current_month:
                    rebalance.add(date)
                    current_month = date.month
            return rebalance
        else:
            return set(dates)

    def walk_forward(
        self,
        sector_returns: pd.DataFrame,
        signals: np.ndarray,
        train_window: int = 252,
        test_window: int = 63,
    ) -> list[dict]:
        """Walk-forward optimization.

        Splits data into rolling train/test windows for
        out-of-sample evaluation.
        """
        results = []
        n_days = len(sector_returns)
        start = 0

        while start + train_window + test_window <= n_days:
            test_start = start + train_window
            test_end = test_start + test_window

            test_returns = sector_returns.iloc[test_start:test_end]
            test_signals = signals[test_start:test_end]

            result = self.run(test_returns, test_signals)
            result["period"] = {
                "train_start": sector_returns.index[start],
                "test_start": sector_returns.index[test_start],
                "test_end": sector_returns.index[min(test_end - 1, n_days - 1)],
            }
            results.append(result)

            start += test_window  # Roll forward

        logger.info(f"Walk-forward: {len(results)} periods evaluated")
        return results
