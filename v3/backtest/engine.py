"""V3 Backtest Engine — stock-level, conditional entry, vol-based exit.

Medallion-inspired enhancements:
  1. Vol-dependent slippage (higher vol → wider spreads → more cost)
  2. Walk-forward validation (train 252d, test 63d, rolling)
  3. Detailed cost breakdown (commission + slippage + spread)
  4. Regime-aware direction evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from v3.config.schema import V3Config
from v3.rules.direction import DirectionEngine
from v3.rules.entry import EntryFilter
from v3.rules.exit import ExitRules
from v3.strategy.risk import RiskManager
from v3.strategy.regime import RegimeDetector


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    ticker: str
    market: str
    direction: str
    entry_price: float
    exit_price: float
    weight: float
    gross_return: float
    net_return: float
    cost: float
    exit_reason: str
    hold_days: int
    vol_score: float
    confidence: float


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    avg_hold_days: float
    total_trades: int
    avg_trades_per_month: float
    total_cost: float
    profit_factor: float
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame


class BacktestEngine:
    """Event-driven backtest engine for V3 vol expansion strategy."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.direction_engine = DirectionEngine(
            momentum_window=cfg.trading.direction_rules.momentum_window,
            momentum_weight=cfg.trading.direction_rules.momentum_weight,
            flow_weight=cfg.trading.direction_rules.flow_weight,
            event_weight=cfg.trading.direction_rules.event_weight,
        )
        self.entry_filter = EntryFilter(
            min_direction_clarity=cfg.trading.direction_rules.min_direction_clarity,
            max_trades_per_month=cfg.trading.max_trades_per_month,
            max_positions=cfg.trading.max_positions,
        )
        self.exit_rules = ExitRules(
            profit_take_pct=cfg.trading.profit_take_pct,
            max_hold_days=cfg.trading.max_hold_days,
            vol_contraction_ratio=cfg.trading.vol_contraction_exit_ratio,
            daily_loss_limit=cfg.risk.daily_loss_limit,
        )
        self.risk_manager = RiskManager(
            daily_loss_limit=cfg.risk.daily_loss_limit,
            total_loss_limit=cfg.risk.total_loss_limit,
        )
        self.regime_detector = RegimeDetector(
            momentum_window=cfg.regime.momentum_window,
            vol_lookback=cfg.regime.vol_lookback,
            bull_threshold=cfg.regime.bull_threshold,
            bear_threshold=cfg.regime.bear_threshold,
        )

    def run(
        self,
        df: pd.DataFrame,
        vol_predictions: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """Run backtest simulation.

        Args:
            df: Full OHLCV + features DataFrame.
            vol_predictions: DataFrame with date, ticker, vol_score, confidence.
            initial_capital: Starting capital (default from config).

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        capital = initial_capital or self.cfg.backtest.initial_capital
        portfolio_value = capital
        cash = capital
        positions: list[dict] = []  # Active positions
        trades: list[TradeRecord] = []
        equity_records = []

        # Get unique trading dates
        pred_dates = sorted(vol_predictions["date"].unique())
        logger.info(f"Backtest: {len(pred_dates)} days, "
                     f"capital={capital:,.0f}, "
                     f"dates: {pred_dates[0]} ~ {pred_dates[-1]}")

        monthly_trades = 0
        current_month = None

        for date in pred_dates:
            date_str = str(date)[:10]
            month = date_str[:7]

            # Reset monthly counter
            if month != current_month:
                current_month = month
                monthly_trades = 0

            # Get today's data
            today_data = df[df["date"] == date]
            if today_data.empty:
                continue

            # Update risk manager
            self.risk_manager.update(portfolio_value)

            # Detect market regime
            market_data = df.groupby("date").agg(close=("close", "mean")).reset_index()
            market_data = market_data[market_data["date"] <= date].sort_values("date")
            regime_state = self.regime_detector.detect(market_data)

            # Step 1: Check exits on existing positions
            new_positions = []
            daily_pnl = 0.0

            for pos in positions:
                ticker_data = today_data[today_data["ticker"] == pos["ticker"]]
                if ticker_data.empty:
                    new_positions.append(pos)
                    continue

                row = ticker_data.iloc[0]
                current_price = row["close"]
                current_vol = row.get("vol_cc_5d", pos.get("entry_vol", 0.3))
                hold_days = pos["hold_days"] + 1

                deployed = 1.0 - (cash / max(portfolio_value, 1))
                exit_decision = self.exit_rules.check(
                    entry_price=pos["entry_price"],
                    current_price=current_price,
                    hold_days=hold_days,
                    entry_vol=pos.get("entry_vol", 0.3),
                    current_vol=current_vol,
                    confidence=pos.get("confidence", 0.5),
                    low_price=row.get("low", current_price),
                    portfolio_deployed=deployed,
                )

                if exit_decision.should_exit:
                    # Close position
                    market = self._market_of(pos["ticker"])
                    cost = self.cfg.cost_for_market(market)
                    slippage = self.get_realistic_slippage(market, current_vol)
                    exit_price = current_price * (1 - slippage)  # Sell slippage

                    gross_ret = exit_price / pos["entry_price"] - 1
                    net_ret = gross_ret - cost

                    trade = TradeRecord(
                        entry_date=pos["entry_date"],
                        exit_date=date_str,
                        ticker=pos["ticker"],
                        market=market,
                        direction="long",
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        weight=pos["weight"],
                        gross_return=gross_ret,
                        net_return=net_ret,
                        cost=cost,
                        exit_reason=exit_decision.reason,
                        hold_days=hold_days,
                        vol_score=pos.get("vol_score", 0),
                        confidence=pos.get("confidence", 0),
                    )
                    trades.append(trade)
                    daily_pnl += net_ret * pos["weight"]
                    cash += pos["capital_allocated"] * (1 + net_ret)
                else:
                    pos["hold_days"] = hold_days
                    # Mark-to-market: daily change in unrealized only
                    unrealized = (current_price / pos["entry_price"] - 1)
                    prev_unrealized = pos.get("last_unrealized", unrealized)  # Day 1: no delta
                    daily_pnl += (unrealized - prev_unrealized) * pos["weight"]
                    pos["last_unrealized"] = unrealized
                    new_positions.append(pos)

            positions = new_positions

            # Step 2: Check for new entries
            today_preds = vol_predictions[vol_predictions["date"] == date]
            circuit_breaker = self.risk_manager.circuit_breaker_scale(portfolio_value) < 1.0

            if not today_preds.empty and not circuit_breaker:
                # Sort by vol_score descending
                sorted_preds = today_preds.sort_values("vol_score", ascending=False)

                for _, pred_row in sorted_preds.iterrows():
                    ticker = pred_row["ticker"]
                    vol_score = pred_row["vol_score"]
                    confidence = pred_row["confidence"]

                    # Skip if already in position
                    if any(p["ticker"] == ticker for p in positions):
                        continue

                    # Get ticker data for direction evaluation
                    ticker_hist = df[(df["ticker"] == ticker) & (df["date"] <= date)]
                    if len(ticker_hist) < 20:
                        continue

                    dir_signal = self.direction_engine.evaluate(
                        ticker, ticker_hist, regime=regime_state.regime,
                    )
                    market = self._market_of(ticker)
                    cost = self.cfg.cost_for_market(market)

                    decision = self.entry_filter.evaluate(
                        ticker=ticker,
                        vol_score=vol_score,
                        confidence=confidence,
                        direction_signal=dir_signal,
                        current_positions=len(positions),
                        monthly_trades=monthly_trades,
                        market=market,
                        cost_rate=cost,
                        circuit_breaker_active=circuit_breaker,
                    )

                    if decision.should_enter:
                        today_row = today_data[today_data["ticker"] == ticker]
                        if today_row.empty:
                            continue

                        row = today_row.iloc[0]
                        ticker_vol = row.get("vol_cc_5d", 0.2)
                        slippage = self.get_realistic_slippage(market, ticker_vol)
                        entry_price = row["open"] * (1 + slippage)  # Buy slippage

                        weight = min(0.95 / max(len(positions) + 1, 1),
                                     self.cfg.trading.max_single_weight)
                        allocated = cash * weight

                        if allocated < self.cfg.trading.min_order_amount_krw:
                            continue

                        # Initial unrealized = close vs entry (for day-1 delta calc)
                        init_unrealized = (row["close"] / entry_price - 1) if entry_price > 0 else 0
                        positions.append({
                            "ticker": ticker,
                            "entry_date": date_str,
                            "entry_price": entry_price,
                            "weight": weight,
                            "hold_days": 0,
                            "entry_vol": row.get("vol_cc_5d", 0.3),
                            "vol_score": vol_score,
                            "confidence": confidence,
                            "capital_allocated": allocated,
                            "last_unrealized": init_unrealized,
                        })
                        cash -= allocated
                        monthly_trades += 1

                    if len(positions) >= self.cfg.trading.max_positions:
                        break

            # Update portfolio value
            invested = sum(
                p["capital_allocated"] * (1 + p.get("last_unrealized", 0))
                for p in positions
            )
            portfolio_value = cash + invested

            equity_records.append({
                "date": date_str,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "n_positions": len(positions),
                "daily_pnl": daily_pnl,
            })

        # Force close remaining positions at end
        for pos in positions:
            last_data = df[df["ticker"] == pos["ticker"]].sort_values("date").iloc[-1]
            market = self._market_of(pos["ticker"])
            cost = self.cfg.cost_for_market(market)
            gross_ret = last_data["close"] / pos["entry_price"] - 1
            net_ret = gross_ret - cost

            trades.append(TradeRecord(
                entry_date=pos["entry_date"],
                exit_date=str(last_data["date"])[:10],
                ticker=pos["ticker"],
                market=market,
                direction="long",
                entry_price=pos["entry_price"],
                exit_price=last_data["close"],
                weight=pos["weight"],
                gross_return=gross_ret,
                net_return=net_ret,
                cost=cost,
                exit_reason="backtest_end",
                hold_days=pos["hold_days"],
                vol_score=pos.get("vol_score", 0),
                confidence=pos.get("confidence", 0),
            ))

        equity_df = pd.DataFrame(equity_records) if equity_records else pd.DataFrame()
        result = self._compute_metrics(trades, equity_df, capital)

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                     f"return={result.total_return:.2%}, "
                     f"Sharpe={result.sharpe:.2f}, "
                     f"MDD={result.max_drawdown:.2%}")

        return result

    def _compute_metrics(
        self, trades: list[TradeRecord], equity_df: pd.DataFrame, capital: float,
    ) -> BacktestResult:
        """Compute backtest performance metrics."""
        if not trades:
            return BacktestResult(
                total_return=0, annual_return=0, sharpe=0, max_drawdown=0,
                win_rate=0, avg_return=0, avg_hold_days=0, total_trades=0,
                avg_trades_per_month=0, total_cost=0, profit_factor=0,
                trades=[], equity_curve=equity_df,
            )

        returns = [t.net_return for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_return = 0
        if not equity_df.empty:
            total_return = equity_df["portfolio_value"].iloc[-1] / capital - 1

        # Days for annualization
        if not equity_df.empty and len(equity_df) > 1:
            n_days = len(equity_df)
            annual_factor = 252 / max(n_days, 1)
        else:
            annual_factor = 1.0

        annual_return = (1 + total_return) ** annual_factor - 1

        # Sharpe from daily returns
        if not equity_df.empty and "daily_pnl" in equity_df.columns:
            daily_returns = equity_df["daily_pnl"].values
            if len(daily_returns) > 5 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # MDD
        if not equity_df.empty:
            peak = equity_df["portfolio_value"].cummax()
            drawdown = (equity_df["portfolio_value"] - peak) / peak
            max_dd = abs(drawdown.min())
        else:
            max_dd = 0.0

        # Trades per month
        if trades:
            first_date = trades[0].entry_date
            last_date = trades[-1].entry_date
            months = max(1, (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days / 30)
            trades_per_month = len(trades) / months
        else:
            trades_per_month = 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-8
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=len(wins) / max(len(trades), 1),
            avg_return=np.mean(returns),
            avg_hold_days=np.mean([t.hold_days for t in trades]),
            total_trades=len(trades),
            avg_trades_per_month=trades_per_month,
            total_cost=sum(t.cost for t in trades),
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_df,
        )

    @staticmethod
    def _market_of(ticker: str) -> str:
        if ticker.endswith(".KS"):
            return "KOSPI"
        elif ticker.endswith(".KQ"):
            return "KOSDAQ"
        elif ticker.split(".")[0].isdigit():
            return "KOSPI"
        return "NASDAQ"

    def get_realistic_slippage(self, market: str, ticker_vol: float = 0.2) -> float:
        """Vol-dependent slippage: higher vol → wider spreads → more cost."""
        base = self.cfg.slippage_for_market(market)
        # Vol adjustment: +50% slippage per 0.1 above baseline (0.2)
        vol_excess = max(0, ticker_vol - 0.20)
        vol_mult = 1.0 + vol_excess * 5.0   # 0.3 vol → 1.5x slippage
        return base * vol_mult

    # ── Walk-Forward Validation ──────────────────────────────

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        vol_predictions: pd.DataFrame,
        train_days: int | None = None,
        test_days: int | None = None,
        step_days: int | None = None,
    ) -> dict:
        """Walk-forward backtest: rolling windows to test robustness.

        Args:
            df: Full OHLCV + features.
            vol_predictions: All vol predictions (full period).
            train_days: Training window size (default from config).
            test_days: Test window size.
            step_days: Step between windows.

        Returns:
            Aggregated walk-forward results.
        """
        train_days = train_days or self.cfg.backtest.walk_forward_train_days
        test_days = test_days or self.cfg.backtest.walk_forward_test_days
        step_days = step_days or self.cfg.backtest.walk_forward_step_days

        dates = sorted(df["date"].unique())
        results = []

        logger.info(f"Walk-forward: train={train_days}d, test={test_days}d, step={step_days}d")

        fold = 0
        for start_idx in range(0, len(dates) - train_days - test_days, step_days):
            test_start = start_idx + train_days
            test_end = test_start + test_days

            test_date_set = set(dates[test_start:test_end])
            test_pred = vol_predictions[vol_predictions["date"].isin(test_date_set)]

            if test_pred.empty:
                continue

            fold += 1
            result = self.run(df, test_pred)

            logger.info(f"  Fold {fold}: {dates[test_start]}~{dates[min(test_end, len(dates)-1)]} "
                         f"trades={result.total_trades} return={result.total_return:.2%} "
                         f"Sharpe={result.sharpe:.2f}")

            results.append(result)

        if not results:
            logger.warning("Walk-forward: no valid folds")
            return {"avg_sharpe": 0, "n_folds": 0}

        agg = self._aggregate_walk_forward(results)

        logger.info("=" * 60)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info(f"  Folds:           {agg['n_folds']}")
        logger.info(f"  Avg Sharpe:      {agg['avg_sharpe']:.2f} ± {agg['std_sharpe']:.2f}")
        logger.info(f"  Avg Return:      {agg['avg_return']:.2%} ± {agg['std_return']:.2%}")
        logger.info(f"  Worst Sharpe:    {agg['worst_sharpe']:.2f}")
        logger.info(f"  Profitable %:    {agg['pct_profitable']:.0%}")
        logger.info(f"  Avg Win Rate:    {agg['avg_win_rate']:.1%}")
        logger.info(f"  Avg Trades/Fold: {agg['avg_trades']:.1f}")
        logger.info("=" * 60)

        return agg

    @staticmethod
    def _aggregate_walk_forward(results: list[BacktestResult]) -> dict:
        returns = [r.total_return for r in results]
        sharpes = [r.sharpe for r in results]
        win_rates = [r.win_rate for r in results]
        trades = [r.total_trades for r in results]

        return {
            "n_folds": len(results),
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "avg_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "worst_sharpe": float(np.min(sharpes)),
            "best_sharpe": float(np.max(sharpes)),
            "pct_profitable": sum(1 for r in returns if r > 0) / max(len(returns), 1),
            "avg_win_rate": float(np.mean(win_rates)),
            "avg_trades": float(np.mean(trades)),
            "results": results,
        }
