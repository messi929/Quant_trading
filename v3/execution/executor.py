"""V3 Trading Executor — vol expansion strategy with conditional entry.

Handles: entry, monitoring, exit, risk management.
Uses V2 KISApi for order execution.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from v3.config.schema import V3Config
from v3.execution.broker import KISApi
from v3.execution.position_manager import PositionManager
from v3.rules.exit import ExitRules
from v3.strategy.risk import RiskManager
from v3.utils.ticker_utils import kis_code, is_domestic, round_to_tick


class TradingExecutor:
    """Executes V3 trades via KIS API."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.api = KISApi()
        self.positions = PositionManager()
        self.risk = RiskManager(
            daily_loss_limit=cfg.risk.daily_loss_limit,
            total_loss_limit=cfg.risk.total_loss_limit,
        )
        self.exit_rules = ExitRules(
            profit_take_pct=cfg.trading.profit_take_pct,
            max_hold_days=cfg.trading.max_hold_days,
            vol_contraction_ratio=cfg.trading.vol_contraction_exit_ratio,
            daily_loss_limit=cfg.risk.daily_loss_limit,
            use_time_decay_tp=True,
            mae_stop_threshold=-0.99 if not cfg.trading.use_mae_stop else -0.03,
            mae_stop_tightened=-0.99 if not cfg.trading.use_mae_stop else -0.025,
        )

    def execute_entry(self, signal: dict, current_date: str) -> list[dict]:
        """Execute entry orders from signal.

        Args:
            signal: TradeSignal-like dict with 'action', 'positions', 'regime'.
            current_date: "YYYY-MM-DD".

        Returns:
            List of executed orders.
        """
        if signal.get("action") == "CASH":
            logger.info("Signal: CASH — closing all positions")
            self.force_close_all("CASH signal")
            return []

        # Circuit breaker check
        balance = self._get_portfolio_value()
        cb_scale = self.risk.circuit_breaker_scale(balance)
        if cb_scale <= 0:
            logger.warning("Circuit breaker: CRISIS — all trading halted")
            self.force_close_all("circuit_breaker_crisis")
            return []

        executed = []
        positions_to_enter = signal.get("positions", [])

        for pos in positions_to_enter:
            ticker = pos["ticker"]
            weight = pos["weight"] * cb_scale
            confidence = pos.get("confidence", 0.5)

            # Cooldown check
            if self.positions.is_cooled_down(ticker, current_date):
                continue

            # Consecutive entry check
            if self.positions.is_consecutive_entry(ticker, current_date):
                continue

            # Skip if already in position
            if self.positions.get_position(ticker):
                continue

            # Monthly limit
            monthly = self.positions.monthly_trade_count(current_date)
            if monthly >= self.cfg.trading.max_trades_per_month:
                logger.info(f"Monthly trade limit reached ({monthly})")
                break

            # Position capacity
            if self.positions.count() >= self.cfg.trading.max_positions:
                break

            # Calculate order size
            order_amount = balance * weight
            if order_amount < self.cfg.trading.min_order_amount_krw:
                continue

            # Execute order
            order = self._place_buy_order(ticker, order_amount)
            if order:
                self.positions.add_position({
                    "ticker": ticker,
                    "entry_date": current_date,
                    "entry_price": order["price"],
                    "weight": weight,
                    "hold_days": 0,
                    "vol_score": pos.get("vol_score", 0),
                    "confidence": confidence,
                    "entry_vol": 0.2,  # Will be updated from market data
                    "capital_allocated": order_amount,
                })
                executed.append(order)
                logger.info(f"BUY {ticker}: {order_amount:,.0f} KRW @ {order['price']}")

        return executed

    def monitor_positions(self, current_date: str) -> list[dict]:
        """Check exits on all open positions.

        Returns list of closed positions.
        """
        closed = []
        portfolio_value = self._get_portfolio_value()
        self.risk.update(portfolio_value)

        for pos in list(self.positions.positions):
            ticker = pos["ticker"]

            # Get current price
            price_info = self._get_price(ticker)
            if not price_info:
                continue

            current_price = price_info["price"]
            hold_days = pos["hold_days"] + 1
            pos["hold_days"] = hold_days

            # Check exit
            exit_decision = self.exit_rules.check(
                entry_price=pos["entry_price"],
                current_price=current_price,
                hold_days=hold_days,
                entry_vol=pos.get("entry_vol", 0.2),
                current_vol=price_info.get("vol", 0.2),
                confidence=pos.get("confidence", 0.5),
                low_price=price_info.get("low", current_price),
            )

            if exit_decision.should_exit:
                order = self._place_sell_order(ticker, pos)
                if order:
                    net_return = (order["price"] / pos["entry_price"] - 1)
                    if net_return < 0:
                        self.positions.record_loss(ticker, current_date)

                    self.positions.remove_position(ticker)
                    closed.append({
                        "ticker": ticker,
                        "reason": exit_decision.reason,
                        "return": net_return,
                        "hold_days": hold_days,
                    })
                    logger.info(f"EXIT {ticker}: {exit_decision.reason} "
                                f"ret={net_return:+.2%} hold={hold_days}d")
                else:
                    # Sell failed — track for ghost removal
                    if self.positions.record_sell_failure(ticker):
                        closed.append({
                            "ticker": ticker,
                            "reason": "ghost_removed",
                            "return": 0,
                            "hold_days": hold_days,
                        })

        self.positions.save()
        return closed

    def force_close_all(self, reason: str = "force_close") -> list[dict]:
        """Close all positions immediately."""
        closed = []
        for pos in list(self.positions.positions):
            ticker = pos["ticker"]
            order = self._place_sell_order(ticker, pos)
            if order:
                net_return = (order["price"] / pos["entry_price"] - 1)
                self.positions.remove_position(ticker)
                closed.append({
                    "ticker": ticker,
                    "reason": reason,
                    "return": net_return,
                })
                logger.info(f"FORCE EXIT {ticker}: {reason} ret={net_return:+.2%}")
            else:
                self.positions.record_sell_failure(ticker)
        return closed

    # ── Private Methods ──────────────────────────────────────

    def _place_buy_order(self, ticker: str, amount: float) -> dict | None:
        """Place a buy order via KIS API."""
        try:
            code = kis_code(ticker)
            if is_domestic(ticker):
                price_info = self.api.get_domestic_price(code)
                price = price_info.get("price", 0)
                if price <= 0:
                    return None
                qty = int(amount / price)
                if qty <= 0:
                    return None
                result = self.api.order_domestic(code, "buy", qty, 0, order_type="01")
                return {"ticker": ticker, "price": price, "qty": qty, "order_no": result.get("order_no")}
            else:
                price_info = self.api.get_overseas_price(code, "NASD")
                price = price_info.get("price", 0)
                if price <= 0:
                    return None
                qty = int(amount / (price * 1400))  # Rough KRW conversion
                if qty <= 0:
                    return None
                result = self.api.order_overseas(code, "buy", qty, price, "NASD")
                return {"ticker": ticker, "price": price, "qty": qty, "order_no": result.get("order_no")}
        except Exception as e:
            logger.error(f"Buy order failed {ticker}: {e}")
            return None

    def _place_sell_order(self, ticker: str, position: dict) -> dict | None:
        """Place a sell order via KIS API."""
        try:
            code = kis_code(ticker)
            qty = position.get("qty", 0)
            if qty <= 0:
                # Try to get qty from balance
                qty = self._get_holding_qty(ticker)
                if qty <= 0:
                    return None

            if is_domestic(ticker):
                result = self.api.order_domestic(code, "sell", qty, 0, order_type="01")
                price_info = self.api.get_domestic_price(code)
                return {"ticker": ticker, "price": price_info.get("price", 0), "qty": qty}
            else:
                price_info = self.api.get_overseas_price(code, "NASD")
                price = price_info.get("price", 0)
                result = self.api.order_overseas(code, "sell", qty, price, "NASD")
                return {"ticker": ticker, "price": price, "qty": qty}
        except Exception as e:
            logger.error(f"Sell order failed {ticker}: {e}")
            return None

    def _get_price(self, ticker: str) -> dict | None:
        """Get current price for a ticker."""
        try:
            code = kis_code(ticker)
            if is_domestic(ticker):
                return self.api.get_domestic_price(code)
            else:
                return self.api.get_overseas_price(code, "NASD")
        except Exception as e:
            logger.debug(f"Price fetch failed {ticker}: {e}")
            return None

    def _get_portfolio_value(self) -> float:
        """Get total portfolio value from broker."""
        try:
            balance = self.api.get_domestic_balance()
            return balance.get("total_eval", 100_000_000)
        except Exception:
            return 100_000_000  # Default

    def _get_holding_qty(self, ticker: str) -> int:
        """Get actual holding quantity from broker balance."""
        try:
            code = kis_code(ticker)
            if is_domestic(ticker):
                balance = self.api.get_domestic_balance()
            else:
                balance = self.api.get_overseas_balance()

            for pos in balance.get("positions", []):
                if kis_code(pos.get("ticker", "")) == code:
                    return int(pos.get("qty", 0))
        except Exception:
            pass
        return 0
