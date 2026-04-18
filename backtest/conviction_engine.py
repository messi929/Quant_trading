"""Conviction-based backtesting engine (v2).

Simulates the full trading cycle:
  Signal → Entry at open → Intraday exit simulation → Track P&L

Uses the SAME ConvictionSignalGenerator as live trading (consistency).
Includes profit-taking, stop-loss, and time-based exits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.intraday_sim import simulate_multi_day_exit, ExitResult
from strategy.stock_signal import ConvictionSignalGenerator


@dataclass
class TradeRecord:
    """Single trade record."""
    date: str
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str
    weight: float
    score: float
    confidence: float
    hold_days: int = 1


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list[TradeRecord]
    daily_returns: pd.Series
    equity_curve: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    n_cash_days: int
    avg_positions: float
    gate_pass: bool


class ConvictionBacktester:
    """Backtests conviction-based stock picking with intraday exits.

    Uses the SAME signal generator as live trading to ensure consistency.
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        commission_rate: float = 0.0002,
        slippage_rate: float = 0.003,
        profit_take_pct: float = 0.025,
        profit_take_full_pct: float = 0.05,
        stop_loss_pct: float = 0.02,
        max_hold_days: int = 3,
        day_trade_default: bool = True,
        min_sharpe: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.profit_take_pct = profit_take_pct
        self.profit_take_full_pct = profit_take_full_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.day_trade_default = False  # v2.3: 3일 보유 전략 — 당일 청산 금지
        self.min_sharpe = min_sharpe

    def run(
        self,
        price_data: pd.DataFrame,
        daily_signals: dict[str, dict],
        market_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            price_data: DataFrame with columns [ticker, date, open, high, low, close, volume].
            daily_signals: {date_str: signal_output} from ConvictionSignalGenerator.
            market_returns: Daily market return series for regime detection.

        Returns:
            BacktestResult with full metrics.
        """
        dates = sorted(daily_signals.keys())
        capital = self.initial_capital
        trades: list[TradeRecord] = []
        daily_returns: list[float] = []
        daily_dates: list[str] = []
        n_cash_days = 0
        position_counts: list[int] = []

        for date_str in dates:
            signal = daily_signals[date_str]

            if signal["action"] == "CASH":
                daily_returns.append(0.0)
                daily_dates.append(date_str)
                n_cash_days += 1
                position_counts.append(0)
                continue

            # Execute trades for this day
            day_return = 0.0
            n_positions = 0

            for pos in signal["positions"]:
                ticker = pos["ticker"]
                weight = pos["weight"]

                # Get price data for this ticker/date
                ticker_data = price_data[
                    (price_data["ticker"] == ticker)
                    & (price_data["date"] >= date_str)
                ].sort_values("date").head(self.max_hold_days + 1)

                if len(ticker_data) < 1:
                    continue

                # Entry at open + slippage
                entry_row = ticker_data.iloc[0]
                open_price = entry_row["open"]
                if open_price <= 0 or pd.isna(open_price):
                    continue

                entry_price = open_price * (1 + self.slippage_rate)

                # Build OHLC list for multi-day simulation
                ohlc_list = []
                for _, row in ticker_data.iterrows():
                    ohlc_list.append({
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                    })

                # Simulate exit
                result = simulate_multi_day_exit(
                    entry_price=entry_price,
                    daily_ohlc=ohlc_list,
                    profit_take_pct=self.profit_take_pct,
                    profit_take_full_pct=self.profit_take_full_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    max_hold_days=self.max_hold_days,
                    day_trade_default=self.day_trade_default,
                )

                # Apply commission + exit slippage (v2.3)
                gross_return = result.return_pct
                cost = self.commission_rate * 2  # buy + sell commission
                exit_slip = self.slippage_rate    # exit slippage (entry already in price)
                net_return = gross_return - cost - exit_slip

                # Weighted contribution to portfolio return (log-additive)
                day_return += net_return * weight
                n_positions += 1

                trades.append(TradeRecord(
                    date=date_str,
                    ticker=ticker,
                    side="long",
                    entry_price=entry_price,
                    exit_price=result.exit_price,
                    return_pct=net_return,
                    exit_reason=result.exit_reason,
                    weight=weight,
                    score=pos["score"],
                    confidence=pos["confidence"],
                    hold_days=min(len(ohlc_list), self.max_hold_days),
                ))

            daily_returns.append(day_return)
            daily_dates.append(date_str)
            position_counts.append(n_positions)

        # Compute metrics
        returns_series = pd.Series(daily_returns, index=pd.to_datetime(daily_dates))
        equity = (1 + returns_series).cumprod() * self.initial_capital

        result = self._compute_metrics(
            trades, returns_series, equity, n_cash_days, position_counts,
        )

        # Gate check
        logger.info("=" * 60)
        logger.info("BACKTEST GATE CHECK")
        logger.info("=" * 60)
        logger.info(f"  Sharpe: {result.sharpe_ratio:.2f} "
                     f"{'PASS' if result.gate_pass else 'FAIL'} "
                     f"(threshold: {self.min_sharpe:.1f})")
        logger.info(f"  Total Return: {result.total_return:.2%}")
        logger.info(f"  Max DD: {result.max_drawdown:.2%}")
        logger.info(f"  Win Rate: {result.win_rate:.1%}")
        logger.info(f"  Trades: {result.n_trades}, Cash Days: {result.n_cash_days}")
        logger.info(f"  Avg Positions/Day: {result.avg_positions:.1f}")
        logger.info("=" * 60)

        return result

    def _compute_metrics(
        self,
        trades: list[TradeRecord],
        returns: pd.Series,
        equity: pd.Series,
        n_cash_days: int,
        position_counts: list[int],
    ) -> BacktestResult:
        """Compute all performance metrics."""
        n_trades = len(trades)

        # Returns
        total_return = float(equity.iloc[-1] / self.initial_capital - 1) if len(equity) > 0 else 0
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe
        daily_mean = returns.mean()
        daily_std = returns.std()
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-8 else 0.0

        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())

        # Trade-level metrics
        if n_trades > 0:
            wins = [t for t in trades if t.return_pct > 0]
            losses = [t for t in trades if t.return_pct <= 0]
            win_rate = len(wins) / n_trades
            avg_win = np.mean([t.return_pct for t in wins]) if wins else 0
            avg_loss = np.mean([t.return_pct for t in losses]) if losses else 0
            gross_profit = sum(t.return_pct for t in wins) if wins else 0
            gross_loss = abs(sum(t.return_pct for t in losses)) if losses else 1e-8
            profit_factor = gross_profit / gross_loss
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0

        avg_positions = np.mean(position_counts) if position_counts else 0.0

        return BacktestResult(
            trades=trades,
            daily_returns=returns,
            equity_curve=equity,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            n_trades=n_trades,
            n_cash_days=n_cash_days,
            avg_positions=avg_positions,
            gate_pass=sharpe >= self.min_sharpe,
        )
