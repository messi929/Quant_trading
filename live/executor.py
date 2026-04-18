"""Trading executor — 3-day hold strategy (v2.3).

Core principles:
  - 모델이 3일 후를 예측 → 3일간 보유하여 예측 실현 기회 확보
  - 개별 종목 스탑로스 제거 → 포트폴리오 레벨 리스크 관리
  - 청산 기준: (1) +5% 이상치 (2) 3일 만료 (3) 신호 반전 (4) 포트폴리오 손실 한도

v2.3 changes (Medallion alignment):
  - 고스트 포지션 강제 제거: 매도 3회 실패 시 open_positions에서 제거
  - 포지션 디스크 영속화: JSON 저장으로 서비스 재시작 시 복구
  - 멀티데이 냉각기: 최근 5일 내 2회 이상 손실 종목 진입 차단
  - 연속 진입 제한: 동일 종목 3일 연속 진입 불가 (신호 정체 방지)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from loguru import logger

from broker.kis_api import KISApi
from tracking.trade_log import TradeLogger
from utils.ticker_utils import kis_code


# Medallion: edge × 반복 × sizing — 반복이 부족하면 edge가 발현되지 않음
# → 동일 종목 반복 진입은 "반복"이 아니라 "집착"
MAX_SELL_RETRIES = 3         # 매도 실패 N회 → 고스트 포지션 강제 제거
COOLDOWN_DAYS = 5            # 냉각기 기간 (일)
COOLDOWN_LOSS_COUNT = 2      # 이 기간 내 N회 손실 → 진입 차단
MAX_CONSECUTIVE_ENTRY = 3    # 동일 종목 연속 N일 진입 → 차단


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

        # Persistence paths
        self._state_dir = Path(config["paths"].get("models", "saved_models"))
        self._positions_path = self._state_dir / "open_positions.json"
        self._cooldown_path = self._state_dir / "ticker_cooldown.json"

        # State (loaded from disk)
        self.open_positions: dict[str, dict] = self._load_positions()
        self._sell_fail_count: dict[str, int] = {}  # ticker → consecutive sell failures
        self._daily_pnl = 0.0
        self._daily_traded_amount = 0.0

        # Multi-day cooldown state (persisted)
        # {ticker_code: [{"date": "2026-04-06", "pnl": -0.03}, ...]}
        self._ticker_history: dict[str, list[dict]] = self._load_cooldown()
        # {ticker_code: ["2026-04-06", "2026-04-07", ...]} — consecutive entry dates
        self._entry_dates: dict[str, list[str]] = {}

    # ── Entry (reconciliation + circuit breaker + cooldown) ────

    def execute_entry(
        self,
        positions: list[dict],
        portfolio_value: Optional[float] = None,
    ) -> list[dict]:
        """Reconcile holdings with signal, trade only the diff.

        1. Circuit breaker scaling (MDD-based)
        2. Close positions NOT in new signal (rebalance)
        3. Keep positions still in signal (save costs)
        4. Cooldown check (multi-day loss history + consecutive entry)
        5. Buy new positions
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
        today_str = date.today().isoformat()
        executed = []
        for pos in positions:
            ticker = pos["ticker"]
            code = kis_code(ticker)
            market = pos.get("market", "domestic")

            if ticker in self.open_positions:
                logger.info(f"KEEP [{ticker}]: already held, skip re-entry")
                continue

            # Multi-day cooldown: 최근 N일 내 M회 손실 → 차단
            if self._is_cooled_down(code):
                logger.info(f"COOLDOWN [{ticker}]: recent losses, skipping")
                continue

            # Consecutive entry limit: 동일 종목 N일 연속 진입 → 차단
            if self._is_consecutive_entry(code, today_str):
                logger.info(
                    f"STALE [{ticker}]: entered {MAX_CONSECUTIVE_ENTRY}+ "
                    f"consecutive days, skipping"
                )
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
                        "entry_time": datetime.now().isoformat(),
                        "weight": weight,
                        "market": market,
                        "score": pos.get("score", 0),
                    }
                    self._record_entry(code, today_str)
                    self._save_positions()
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

            reason = None

            # +5% 이상치 → 조기 청산
            if pnl_pct >= self.profit_take_full_pct:
                reason = "profit_take_full"

            # 3일 보유 만료 → 예측 기간 종료
            elif self._hold_days(pos) >= self.max_hold_days:
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
            if self._hold_days(pos) >= self.max_hold_days:
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
            self._sell_fail_count.pop(ticker, None)
            self._save_positions()

            # Record loss for cooldown
            code = kis_code(ticker)
            self._record_trade_result(code, pnl_pct)

            entry_time = pos.get("entry_time", "")
            if isinstance(entry_time, str):
                try:
                    hold = (datetime.now() - datetime.fromisoformat(entry_time)).days
                except ValueError:
                    hold = 0
            else:
                hold = (datetime.now() - entry_time).days

            sign = "+" if pnl_pct > 0 else ""
            logger.info(
                f"EXIT [{reason}]: {ticker} {qty}주 @ {price:,.0f} "
                f"({sign}{pnl_pct:.2%}, {hold}d held)"
            )
        else:
            # Sell failed — track retries and force-remove ghosts
            self._handle_sell_failure(ticker, pos)

        return result

    # ── Sell failure recovery (ghost position elimination) ────

    def _handle_sell_failure(self, ticker: str, pos: dict):
        """Track sell failures. Force-remove after MAX_SELL_RETRIES.

        Medallion principle: infrastructure failures must not block trading.
        A position that can't be sold doesn't exist — remove it.
        """
        self._sell_fail_count[ticker] = self._sell_fail_count.get(ticker, 0) + 1
        count = self._sell_fail_count[ticker]

        if count >= MAX_SELL_RETRIES:
            logger.warning(
                f"GHOST REMOVAL: {ticker} sell failed {count} times "
                f"— force-removing from open_positions"
            )
            self.open_positions.pop(ticker, None)
            self._sell_fail_count.pop(ticker, None)
            self._save_positions()
            return

        # Try balance verification (may also fail on sandbox)
        if pos.get("market") == "domestic":
            self._verify_and_sync(ticker)

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
                self._sell_fail_count.pop(ticker, None)
                self._save_positions()
            else:
                logger.info(
                    f"SYNC: {ticker} confirmed in balance — "
                    f"retry {self._sell_fail_count.get(ticker, 0)}/{MAX_SELL_RETRIES}"
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

    def _hold_days(self, pos: dict) -> int:
        """Calculate hold days from entry_time (str or datetime)."""
        entry = pos.get("entry_time")
        if entry is None:
            return 0
        if isinstance(entry, str):
            try:
                entry = datetime.fromisoformat(entry)
            except ValueError:
                return 0
        return (datetime.now() - entry).days

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
        self._daily_pnl = 0.0
        self._daily_traded_amount = 0.0
        # Note: cooldown is multi-day, NOT reset daily (unlike v2.2 _stoploss_tickers)

    # ── Multi-day cooldown (Medallion: avoid repeated losses) ─

    def _is_cooled_down(self, code: str) -> bool:
        """Check if ticker is in cooldown (recent losses exceed threshold)."""
        history = self._ticker_history.get(code, [])
        if not history:
            return False

        cutoff = date.today().isoformat()
        recent_losses = 0
        for entry in history:
            days_ago = (date.today() - date.fromisoformat(entry["date"])).days
            if days_ago <= COOLDOWN_DAYS and entry["pnl"] < 0:
                recent_losses += 1

        return recent_losses >= COOLDOWN_LOSS_COUNT

    def _is_consecutive_entry(self, code: str, today_str: str) -> bool:
        """Check if ticker has been entered too many consecutive days.

        Prevents "signal staleness" — same top pick for days means
        the model is insensitive to recent price changes, not "convicted".
        """
        dates = self._entry_dates.get(code, [])
        if len(dates) < MAX_CONSECUTIVE_ENTRY:
            return False

        # Check if last N entries are consecutive calendar days
        recent = sorted(dates)[-MAX_CONSECUTIVE_ENTRY:]
        for i in range(1, len(recent)):
            d1 = date.fromisoformat(recent[i - 1])
            d2 = date.fromisoformat(recent[i])
            # Allow weekend gaps (max 3 days between consecutive trading days)
            if (d2 - d1).days > 3:
                return False
        return True

    def _record_entry(self, code: str, today_str: str):
        """Record entry date for consecutive-entry tracking."""
        if code not in self._entry_dates:
            self._entry_dates[code] = []
        if today_str not in self._entry_dates[code]:
            self._entry_dates[code].append(today_str)
        # Keep only recent entries
        self._entry_dates[code] = self._entry_dates[code][-10:]

    def _record_trade_result(self, code: str, pnl_pct: float):
        """Record trade P&L for cooldown tracking (persisted)."""
        if code not in self._ticker_history:
            self._ticker_history[code] = []
        self._ticker_history[code].append({
            "date": date.today().isoformat(),
            "pnl": round(pnl_pct, 4),
        })
        # Keep only recent history
        self._ticker_history[code] = self._ticker_history[code][-20:]
        self._save_cooldown()

    # ── Position persistence (survive restarts) ───────────────

    def _save_positions(self):
        """Save open positions to disk."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            # Convert datetime to string for JSON
            data = {}
            for ticker, pos in self.open_positions.items():
                entry = dict(pos)
                if isinstance(entry.get("entry_time"), datetime):
                    entry["entry_time"] = entry["entry_time"].isoformat()
                data[ticker] = entry
            self._positions_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Position save failed: {e}")

    def _load_positions(self) -> dict[str, dict]:
        """Load open positions from disk."""
        try:
            if self._positions_path.exists():
                data = json.loads(self._positions_path.read_text())
                if isinstance(data, dict) and data:
                    logger.info(
                        f"Positions restored: {len(data)} from {self._positions_path}"
                    )
                    return data
        except Exception as e:
            logger.warning(f"Position load failed: {e}")
        return {}

    def _save_cooldown(self):
        """Save cooldown history to disk."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._cooldown_path.write_text(
                json.dumps(self._ticker_history, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Cooldown save failed: {e}")

    def _load_cooldown(self) -> dict[str, list[dict]]:
        """Load cooldown history from disk."""
        try:
            if self._cooldown_path.exists():
                data = json.loads(self._cooldown_path.read_text())
                if isinstance(data, dict):
                    logger.info(f"Cooldown history loaded: {len(data)} tickers")
                    return data
        except Exception as e:
            logger.warning(f"Cooldown load failed: {e}")
        return {}
