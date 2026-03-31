"""Trading executor — entry, monitor, exit (v2).

Simple execution loop:
  1. execute_entry(): Buy positions from signal
  2. monitor_positions(): Check profit/stop/time exits every 30min
  3. execute_exit(): Close all remaining positions at session end

Replaces v1 signal_to_order.py (1700+ lines → ~300 lines).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from broker.kis_api import KISApi
from tracking.trade_log import TradeLogger
from utils.ticker_utils import kis_code, is_domestic, kospi_tick_size


class TradingExecutor:
    """Executes trades based on ConvictionSignalGenerator output."""

    def __init__(
        self,
        config: dict,
        api: Optional[KISApi] = None,
        trade_logger: Optional[TradeLogger] = None,
    ):
        cfg_trading = config["trading"]
        cfg_broker = config["broker"]
        cfg_risk = config["risk"]

        self.profit_take_pct = cfg_trading["profit_take_pct"]
        self.profit_take_full_pct = cfg_trading["profit_take_full_pct"]
        self.stop_loss_pct = cfg_trading["stop_loss_pct"]
        self.time_exit_hours = cfg_trading["time_exit_hours"]
        self.time_exit_min_profit = cfg_trading["time_exit_min_profit"]
        self.max_hold_days = cfg_trading["max_hold_days"]
        self.retry_wait_secs = cfg_trading["retry_wait_secs"]
        self.daily_loss_limit = cfg_risk["daily_loss_limit"]
        self.cost_rate = cfg_trading["transaction_cost_rate"]
        self.volume_max_pct = cfg_trading["volume_participation_max"]

        self.paper_trading = cfg_broker.get("paper_trading", True)
        self.mode = cfg_broker.get("mode", "sandbox")

        if api is None:
            self.api = KISApi(mode=self.mode, market_type="domestic")
        else:
            self.api = api

        if trade_logger is None:
            self.logger = TradeLogger(config["paths"]["trade_log_db"])
        else:
            self.logger = trade_logger

        # Active positions: {ticker: {qty, entry_price, entry_time, weight, ...}}
        self.open_positions: dict[str, dict] = {}

        # Stoploss cooldown (no rebuy same day)
        self._stoploss_tickers: set[str] = set()

        # Daily P&L tracking
        self._daily_pnl = 0.0

    # ── Entry ──────────────────────────────────────────────────

    def execute_entry(
        self,
        positions: list[dict],
        portfolio_value: Optional[float] = None,
    ) -> list[dict]:
        """Execute buy orders for signal positions.

        Args:
            positions: From ConvictionSignalGenerator.generate()["positions"]
            portfolio_value: Total portfolio value for sizing. If None, queries API.

        Returns:
            List of executed order results.
        """
        if not positions:
            return []

        if portfolio_value is None:
            try:
                balance = self.api.get_domestic_balance()
                portfolio_value = balance.get("total_eval", 100_000_000)
            except Exception as e:
                logger.error(f"Balance query failed: {e}")
                return []

        executed = []
        for pos in positions:
            ticker = pos["ticker"]
            weight = pos["weight"]
            market = pos.get("market", "domestic")

            # Cooldown check
            if kis_code(ticker) in self._stoploss_tickers:
                logger.info(f"Skip [{ticker}]: stoploss cooldown")
                continue

            # Daily loss check
            if self._daily_pnl <= -self.daily_loss_limit:
                logger.warning(f"Daily loss limit reached ({self._daily_pnl:.2%})")
                break

            try:
                amount = portfolio_value * weight
                price = self._get_price(ticker, market)
                if price <= 0:
                    continue

                qty = int(amount / price)
                if qty < 1:
                    continue

                # Execute
                result = self._place_order(ticker, "buy", qty, price, market)
                if result:
                    self.open_positions[ticker] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_time": datetime.now(),
                        "weight": weight,
                        "market": market,
                        "score": pos.get("score", 0),
                    }
                    self.logger.log_trade(result, note="entry")
                    executed.append(result)
                    logger.info(
                        f"ENTRY: {ticker} {qty}주 @ {price:,.0f} "
                        f"(weight={weight:.0%}, score={pos.get('score', 0):.4f})"
                    )

            except Exception as e:
                logger.error(f"Entry failed [{ticker}]: {e}")

        return executed

    # ── Monitor ────────────────────────────────────────────────

    def monitor_positions(self) -> list[dict]:
        """Check all open positions for exit conditions.

        Called every 30 minutes during trading session.

        Returns:
            List of exit order results.
        """
        exits = []

        for ticker, pos in list(self.open_positions.items()):
            try:
                current_price = self._get_price(ticker, pos["market"])
                if current_price <= 0:
                    continue

                entry_price = pos["entry_price"]
                pnl_pct = (current_price - entry_price) / entry_price
                elapsed = datetime.now() - pos["entry_time"]
                elapsed_hours = elapsed.total_seconds() / 3600

                exit_reason = None

                # Stop loss: -2%
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = "stop_loss"

                # Profit take (full): +5%
                elif pnl_pct >= self.profit_take_full_pct:
                    exit_reason = "profit_take_full"

                # Profit take (partial): +2.5%
                elif pnl_pct >= self.profit_take_pct:
                    exit_reason = "profit_take"

                # Time exit: 2h and < +1%
                elif (elapsed_hours >= self.time_exit_hours
                      and pnl_pct < self.time_exit_min_profit):
                    exit_reason = "time_exit"

                if exit_reason:
                    result = self._close_position(ticker, pos, current_price, exit_reason)
                    if result:
                        exits.append(result)

            except Exception as e:
                logger.error(f"Monitor error [{ticker}]: {e}")

        return exits

    # ── Session Close ──────────────────────────────────────────

    def execute_exit(self) -> list[dict]:
        """Close ALL remaining positions (session end).

        Returns:
            List of exit order results.
        """
        exits = []
        for ticker, pos in list(self.open_positions.items()):
            try:
                current_price = self._get_price(ticker, pos["market"])
                if current_price <= 0:
                    current_price = pos["entry_price"]  # Fallback

                result = self._close_position(ticker, pos, current_price, "session_close")
                if result:
                    exits.append(result)
            except Exception as e:
                logger.error(f"Exit failed [{ticker}]: {e}")

        logger.info(f"Session close: {len(exits)} positions exited")
        return exits

    # ── Internal ───────────────────────────────────────────────

    def _close_position(
        self,
        ticker: str,
        pos: dict,
        current_price: float,
        reason: str,
    ) -> dict | None:
        """Close a single position."""
        qty = pos["qty"]
        result = self._place_order(ticker, "sell", qty, current_price, pos["market"])

        if result:
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            self._daily_pnl += pnl_pct * pos["weight"]

            self.logger.log_trade(result, note=reason)
            del self.open_positions[ticker]

            if reason == "stop_loss":
                self._stoploss_tickers.add(kis_code(ticker))

            symbol = "+" if pnl_pct > 0 else ""
            logger.info(
                f"EXIT [{reason}]: {ticker} {qty}주 @ {current_price:,.0f} "
                f"({symbol}{pnl_pct:.2%})"
            )

        return result

    def _get_price(self, ticker: str, market: str) -> float:
        """Get current price from API or yfinance fallback."""
        try:
            if market == "domestic":
                data = self.api.get_domestic_price(kis_code(ticker))
                return float(data["price"])
            else:
                if self.paper_trading:
                    import yfinance as yf
                    return float(yf.Ticker(ticker).fast_info.last_price)
                else:
                    data = self.api.get_overseas_price(ticker, "NAS")
                    return float(data["price"])
        except Exception as e:
            logger.debug(f"Price fetch failed [{ticker}]: {e}")
            return 0.0

    def _place_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        price: float,
        market: str,
    ) -> dict | None:
        """Place order via KIS API or paper trade."""
        try:
            if self.paper_trading and market == "overseas":
                # Paper trade for overseas
                return {
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "amount": qty * price,
                    "order_no": f"PAPER-{int(time.time())}",
                    "market": market,
                }

            code = kis_code(ticker)
            if side == "buy":
                result = self.api.order_domestic(
                    ticker=code, qty=qty, price=int(price), side="buy",
                )
            else:
                result = self.api.order_domestic(
                    ticker=code, qty=qty, price=int(price), side="sell",
                )

            return {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": price,
                "amount": qty * price,
                "order_no": result.get("order_no", ""),
                "market": market,
            }

        except Exception as e:
            logger.error(f"Order failed [{side} {ticker} x{qty}]: {e}")
            return None

    def reset_daily(self):
        """Reset daily state (call at start of each trading day)."""
        self._stoploss_tickers.clear()
        self._daily_pnl = 0.0
        self.open_positions.clear()
