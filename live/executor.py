"""Trading executor — 3-day hold strategy (v2.2).

Core principles:
  - 모델이 3일 후를 예측 → 3일간 보유하여 예측 실현 기회 확보
  - 개별 종목 스탑로스 제거 → 포트폴리오 레벨 리스크 관리
  - 청산 기준: (1) +5% 이상치 (2) 3일 만료 (3) 신호 반전 (4) 포트폴리오 손실 한도

v2.2 changes:
  - 개별 stop_loss 제거 (노이즈 청산 방지)
  - time_exit 제거 (alpha decay 기간까지 보유)
  - 부분 이익실현 제거 (+5% 전량만)
  - session_close: 만료 포지션만 청산, 나머지 오버나이트 보유
  - 포트폴리오 daily_loss_limit: 미실현 P&L 포함
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from loguru import logger

from broker.kis_api import KISApi
from tracking.trade_log import TradeLogger
from utils.ticker_utils import kis_code


class TradingExecutor:
    """Executes trades with 3-day hold strategy."""

    def __init__(
        self,
        config: dict,
        api: Optional[KISApi] = None,
        trade_logger: Optional[TradeLogger] = None,
    ):
        trading = config["trading"]
        broker = config["broker"]
        risk = config["risk"]

        # Exit: +5% 이상치에서만 조기 청산
        self.profit_take_full_pct = trading["profit_take_full_pct"]
        # 3일 보유 후 만료 청산
        self.max_hold_days = trading["max_hold_days"]
        self.cost_rate = trading["transaction_cost_rate"]

        # 포트폴리오 레벨 리스크
        self.daily_loss_limit = risk["daily_loss_limit"]
        self.circuit_breaker_cfg = risk["circuit_breaker"]

        # Broker
        self.paper_trading = broker.get("paper_trading", True)
        self.mode = broker.get("mode", "sandbox")

        self.api = api or KISApi(mode=self.mode, market_type="domestic")
        self.trade_logger = trade_logger or TradeLogger(config["paths"]["trade_log_db"])

        # State
        self.open_positions: dict[str, dict] = {}
        self._stoploss_tickers: set[str] = set()
        self._daily_pnl = 0.0
        self._daily_traded_amount = 0.0

    # ── Entry (reconciliation + circuit breaker) ──────────────

    def execute_entry(
        self,
        positions: list[dict],
        portfolio_value: Optional[float] = None,
    ) -> list[dict]:
        """Reconcile holdings with signal, trade only the diff.

        1. Circuit breaker scaling (MDD-based)
        2. Close positions NOT in new signal (rebalance)
        3. Keep positions still in signal (save costs)
        4. Buy new positions
        """
        if portfolio_value is None:
            portfolio_value = self._get_portfolio_value()
            if portfolio_value is None:
                return []

        # Circuit breaker
        cb_scale = self._circuit_breaker_scale()
        if cb_scale <= 0:
            logger.warning("CIRCUIT BREAKER [crisis]: all trading halted")
            return self.force_close_all()
        if cb_scale < 1.0:
            logger.warning(f"CIRCUIT BREAKER: scaling positions to {cb_scale:.0%}")

        # Reconcile: close positions not in new signal
        target_tickers = {p["ticker"] for p in positions}
        for ticker in list(self.open_positions):
            if ticker not in target_tickers:
                pos = self.open_positions[ticker]
                price = self._get_price(ticker, pos["market"])
                if price > 0:
                    self._close_position(ticker, pos, price, "rebalance")

        # Buy new positions
        executed = []
        for pos in positions:
            ticker = pos["ticker"]
            market = pos.get("market", "domestic")

            if ticker in self.open_positions:
                logger.info(f"KEEP [{ticker}]: already held, skip re-entry")
                continue

            if kis_code(ticker) in self._stoploss_tickers:
                logger.info(f"SKIP [{ticker}]: cooldown")
                continue

            # Portfolio loss check (realized only, for entry gating)
            if self._daily_pnl <= -self.daily_loss_limit:
                logger.warning(f"STOP: daily loss limit ({self._daily_pnl:.2%})")
                break

            try:
                weight = pos["weight"] * cb_scale
                price = self._get_price(ticker, market)
                if price <= 0:
                    continue

                qty = int(portfolio_value * weight / price)
                if qty < 1:
                    continue

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
                    self.trade_logger.log_trade(result, note="entry")
                    self._daily_traded_amount += qty * price
                    executed.append(result)
                    logger.info(
                        f"ENTRY: {ticker} {qty}주 @ {price:,.0f} "
                        f"(weight={weight:.0%}, score={pos.get('score', 0):.4f})"
                    )
            except Exception as e:
                logger.error(f"Entry failed [{ticker}]: {e}")

        return executed

    # ── Monitor (portfolio-level risk) ────────────────────────

    def monitor_positions(self) -> list[dict]:
        """Check exit conditions every 5 minutes.

        Exit conditions (no individual stop loss):
          1. Portfolio daily loss limit (realized + unrealized)
          2. Profit take full (+5%) → close
          3. Hold expired (3 days) → close
        """
        if not self.open_positions:
            return []

        # Fetch all prices and calculate unrealized P&L
        unrealized_pnl = 0.0
        position_data: dict[str, tuple[float, float]] = {}

        for ticker, pos in self.open_positions.items():
            price = self._get_price(ticker, pos["market"])
            if price <= 0:
                continue
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            position_data[ticker] = (price, pnl_pct)
            unrealized_pnl += pnl_pct * pos["weight"]

        # Portfolio-level risk: realized + unrealized
        total_pnl = self._daily_pnl + unrealized_pnl
        if total_pnl <= -self.daily_loss_limit:
            logger.warning(
                f"PORTFOLIO RISK: daily P&L {total_pnl:.2%} "
                f"(realized={self._daily_pnl:.2%}, unrealized={unrealized_pnl:.2%}) "
                f"→ closing all positions"
            )
            return self.force_close_all()

        # Individual position checks
        exits = []
        for ticker, pos in list(self.open_positions.items()):
            if ticker not in position_data:
                continue

            price, pnl_pct = position_data[ticker]
            hold_days = (datetime.now() - pos["entry_time"]).days

            reason = None

            # +5% 이상치 → 조기 청산
            if pnl_pct >= self.profit_take_full_pct:
                reason = "profit_take_full"

            # 3일 보유 만료 → 예측 기간 종료
            elif hold_days >= self.max_hold_days:
                reason = "hold_expired"

            if reason:
                result = self._close_position(ticker, pos, price, reason)
                if result:
                    exits.append(result)

        return exits

    # ── Session Close (non-destructive) ───────────────────────

    def execute_exit(self) -> list[dict]:
        """Session close: only close expired positions. Keep others overnight.

        3일 미만 보유 포지션은 오버나이트 유지 (턴오버 감소).
        """
        exits = []
        for ticker, pos in list(self.open_positions.items()):
            hold_days = (datetime.now() - pos["entry_time"]).days
            if hold_days >= self.max_hold_days:
                price = self._get_price(ticker, pos["market"])
                if price <= 0:
                    price = pos["entry_price"]
                result = self._close_position(ticker, pos, price, "hold_expired")
                if result:
                    exits.append(result)

        kept = len(self.open_positions)
        logger.info(
            f"Session close: {len(exits)} expired, {kept} carried overnight"
        )
        return exits

    def force_close_all(self) -> list[dict]:
        """Emergency: close ALL positions (crisis, CASH signal)."""
        exits = []
        for ticker, pos in list(self.open_positions.items()):
            price = self._get_price(ticker, pos["market"])
            if price <= 0:
                price = pos["entry_price"]
            result = self._close_position(ticker, pos, price, "force_close")
            if result:
                exits.append(result)
        logger.info(f"Force close: {len(exits)} positions closed")
        return exits

    # ── Position close ────────────────────────────────────────

    def _close_position(
        self,
        ticker: str,
        pos: dict,
        price: float,
        reason: str,
    ) -> dict | None:
        """Close a position fully."""
        qty = pos["qty"]
        result = self._place_order(ticker, "sell", qty, price, pos["market"])

        if result:
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            self._daily_pnl += pnl_pct * pos["weight"]
            self._daily_traded_amount += qty * price

            self.trade_logger.log_trade(result, note=reason)
            del self.open_positions[ticker]

            if reason in ("force_close", "rebalance"):
                self._stoploss_tickers.add(kis_code(ticker))

            sign = "+" if pnl_pct > 0 else ""
            hold = (datetime.now() - pos["entry_time"]).days
            logger.info(
                f"EXIT [{reason}]: {ticker} {qty}주 @ {price:,.0f} "
                f"({sign}{pnl_pct:.2%}, {hold}d held)"
            )
        else:
            if pos["market"] == "domestic":
                self._verify_and_sync(ticker)

        return result

    # ── Sell failure recovery ─────────────────────────────────

    def _verify_and_sync(self, ticker: str):
        """On sell failure, check actual KIS balance and sync."""
        try:
            balance = self.api.get_domestic_balance()
            held_codes = {
                p.get("ticker", "") for p in balance.get("positions", [])
            }
            code = kis_code(ticker)

            if code not in held_codes:
                logger.warning(
                    f"SYNC: {ticker} ({code}) not in KIS balance — "
                    f"removing from open_positions"
                )
                self.open_positions.pop(ticker, None)
            else:
                logger.info(
                    f"SYNC: {ticker} confirmed in balance — retry next cycle"
                )
        except Exception as e:
            logger.error(f"Balance verification failed: {e}")

    # ── Circuit breaker ───────────────────────────────────────

    def _circuit_breaker_scale(self) -> float:
        """MDD-based position scaling."""
        mdd = abs(self.trade_logger._compute_cumulative_mdd())
        cb = self.circuit_breaker_cfg

        if mdd >= cb["crisis"]:
            return 0.0
        if mdd >= cb["high"]:
            return 0.25
        if mdd >= cb["warning"]:
            return 0.50
        if mdd >= cb["caution"]:
            return 0.75
        return 1.0

    # ── Helpers ────────────────────────────────────────────────

    def _get_portfolio_value(self) -> float | None:
        try:
            balance = self.api.get_domestic_balance()
            return balance.get("total_eval", 100_000_000)
        except Exception as e:
            logger.error(f"Balance query failed: {e}")
            return None

    def _get_price(self, ticker: str, market: str) -> float:
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
        try:
            if self.paper_trading and market == "overseas":
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
            result = self.api.order_domestic(
                ticker=code, qty=qty, price=int(price), side=side,
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
        """Reset daily counters. Preserves open_positions for carry-over."""
        self._stoploss_tickers.clear()
        self._daily_pnl = 0.0
        self._daily_traded_amount = 0.0
