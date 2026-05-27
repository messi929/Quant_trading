"""Paper trading broker — simulates orders using yfinance live prices.

KIS sandbox doesn't support overseas stocks. This broker enables
full NASDAQ testing locally with real market prices.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yfinance as yf
from loguru import logger


class PaperBroker:
    """Simulates brokerage with real prices from yfinance."""

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        save_dir: str = "v3/saved_models",
        commission_rate: float = 0.0001,    # US 편도 0.01%
        slippage_rate: float = 0.001,        # US 0.1%
        usd_krw: float = 1400.0,             # A6 (2026-05-27): fallback only; 실시간은 _fetch_fx_rate()
    ):
        self.commission = commission_rate
        self.slippage = slippage_rate
        self.usd_krw = usd_krw                # 마지막 fetched rate (cache)
        self._fx_fallback = usd_krw           # fetch 실패 시 사용
        self.save_path = Path(save_dir) / "paper_account.json"

        # Load or init account
        if self.save_path.exists():
            self._load()
        else:
            self.cash_krw = initial_capital
            self.positions: list[dict] = []
            self.trade_history: list[dict] = []
            self._save()

        # A6: 초기 환율 실시간 fetch (실패 시 fallback)
        self._fetch_fx_rate()

        logger.info(f"PaperBroker: cash={self.cash_krw:,.0f} KRW, "
                     f"positions={len(self.positions)}, rate={self.usd_krw:.2f} KRW/USD")

    def _fetch_fx_rate(self) -> float:
        """A6 (2026-05-27): yfinance에서 USD/KRW 실시간 환율 fetch.

        실패 시 self._fx_fallback (init 시 받은 1400 기본) 사용. self.usd_krw를
        성공 시 update하여 다음 호출까지 cache. paper PnL accuracy 개선
        (subagent B finding: 고정 1400은 paper PnL ±2%p 오차 유발).
        """
        try:
            t = yf.Ticker("KRW=X")
            info = t.fast_info
            rate = float(info.get("lastPrice", 0) or info.get("previousClose", 0))
            if rate <= 0:
                hist = t.history(period="1d")
                if not hist.empty:
                    rate = float(hist["Close"].iloc[-1])
            if rate > 0 and 800.0 < rate < 2000.0:   # sanity range
                self.usd_krw = rate
                return rate
        except Exception as e:
            logger.warning(f"FX fetch failed (KRW=X): {e} — fallback {self._fx_fallback}")
        return self._fx_fallback

    def get_price(self, ticker: str) -> dict:
        """Get current price from yfinance + fresh FX rate (A6)."""
        # A6: 매 가격 fetch마다 환율도 fresh
        self._fetch_fx_rate()
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = float(info.get("lastPrice", 0) or info.get("previousClose", 0))
            if price <= 0:
                # Fallback: get from history
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            return {
                "ticker": ticker,
                "price": price,
                "price_krw": price * self.usd_krw,
            }
        except Exception as e:
            logger.warning(f"Price fetch failed {ticker}: {e}")
            return {"ticker": ticker, "price": 0, "price_krw": 0}

    def buy(self, ticker: str, amount_krw: float) -> dict | None:
        """Buy stock with KRW amount.

        Args:
            ticker: US ticker (e.g., "AAPL")
            amount_krw: Order amount in KRW

        Returns:
            Order result dict or None if failed.
        """
        price_info = self.get_price(ticker)
        price_usd = price_info["price"]
        if price_usd <= 0:
            logger.error(f"Cannot buy {ticker}: price=0")
            return None

        # Apply slippage (buy higher)
        exec_price = price_usd * (1 + self.slippage)
        exec_price_krw = exec_price * self.usd_krw

        # Calculate quantity
        qty = int(amount_krw / exec_price_krw)
        if qty <= 0:
            logger.warning(f"Cannot buy {ticker}: amount {amount_krw:,.0f} < 1 share ({exec_price_krw:,.0f})")
            return None

        # Cost
        total_cost_krw = qty * exec_price_krw
        commission_krw = total_cost_krw * self.commission

        if total_cost_krw + commission_krw > self.cash_krw:
            logger.warning(f"Insufficient cash: need {total_cost_krw:,.0f}, have {self.cash_krw:,.0f}")
            return None

        # Execute
        self.cash_krw -= (total_cost_krw + commission_krw)
        self.positions.append({
            "ticker": ticker,
            "qty": qty,
            "entry_price_usd": exec_price,
            "entry_price_krw": exec_price_krw,
            "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "amount_krw": total_cost_krw,
        })

        trade = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "BUY",
            "ticker": ticker,
            "qty": qty,
            "price_usd": exec_price,
            "amount_krw": total_cost_krw,
            "commission_krw": commission_krw,
        }
        self.trade_history.append(trade)
        self._save()

        logger.info(f"PAPER BUY {ticker}: {qty}주 @ ${exec_price:.2f} "
                     f"= {total_cost_krw:,.0f} KRW (수수료 {commission_krw:,.0f})")
        return trade

    def sell(self, ticker: str, qty: int | None = None) -> dict | None:
        """Sell stock.

        Args:
            ticker: US ticker
            qty: Number of shares (None = sell all)
        """
        # Find position
        pos = None
        pos_idx = None
        for i, p in enumerate(self.positions):
            if p["ticker"] == ticker:
                pos = p
                pos_idx = i
                break

        if pos is None:
            logger.warning(f"No position for {ticker}")
            return None

        sell_qty = qty or pos["qty"]
        if sell_qty > pos["qty"]:
            sell_qty = pos["qty"]

        price_info = self.get_price(ticker)
        price_usd = price_info["price"]
        if price_usd <= 0:
            logger.error(f"Cannot sell {ticker}: price=0")
            return None

        # Apply slippage (sell lower)
        exec_price = price_usd * (1 - self.slippage)
        exec_price_krw = exec_price * self.usd_krw
        total_krw = sell_qty * exec_price_krw
        commission_krw = total_krw * self.commission

        # P&L
        entry_value = sell_qty * pos["entry_price_krw"]
        pnl_krw = total_krw - entry_value - commission_krw
        pnl_pct = (exec_price / pos["entry_price_usd"] - 1)

        # Update cash
        self.cash_krw += (total_krw - commission_krw)

        # Update or remove position
        if sell_qty >= pos["qty"]:
            self.positions.pop(pos_idx)
        else:
            self.positions[pos_idx]["qty"] -= sell_qty

        trade = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "SELL",
            "ticker": ticker,
            "qty": sell_qty,
            "price_usd": exec_price,
            "amount_krw": total_krw,
            "commission_krw": commission_krw,
            "pnl_krw": pnl_krw,
            "pnl_pct": pnl_pct,
        }
        self.trade_history.append(trade)
        self._save()

        logger.info(f"PAPER SELL {ticker}: {sell_qty}주 @ ${exec_price:.2f} "
                     f"= {total_krw:,.0f} KRW, P&L: {pnl_krw:+,.0f} ({pnl_pct:+.2%})")
        return trade

    def get_balance(self) -> dict:
        """Get current account balance with mark-to-market."""
        total_position_value = 0
        position_details = []

        for pos in self.positions:
            price_info = self.get_price(pos["ticker"])
            current_price = price_info["price"]
            current_value = pos["qty"] * current_price * self.usd_krw
            entry_value = pos["qty"] * pos["entry_price_krw"]
            pnl = current_value - entry_value
            pnl_pct = (current_price / pos["entry_price_usd"] - 1) if pos["entry_price_usd"] > 0 else 0

            total_position_value += current_value
            position_details.append({
                "ticker": pos["ticker"],
                "qty": pos["qty"],
                "entry_price": pos["entry_price_usd"],
                "current_price": current_price,
                "value_krw": current_value,
                "pnl_krw": pnl,
                "pnl_pct": pnl_pct,
                "entry_date": pos["entry_date"],
            })

        total = self.cash_krw + total_position_value
        return {
            "cash": self.cash_krw,
            "position_value": total_position_value,
            "total_eval": total,
            "positions": position_details,
            "n_trades": len(self.trade_history),
        }

    def print_summary(self) -> None:
        """Print account summary."""
        bal = self.get_balance()
        print("\n" + "=" * 55)
        print("V3 Paper Trading Account")
        print("=" * 55)
        print(f"  Cash:       {bal['cash']:>15,.0f} KRW")
        print(f"  Positions:  {bal['position_value']:>15,.0f} KRW")
        print(f"  Total:      {bal['total_eval']:>15,.0f} KRW")
        print(f"  Trades:     {bal['n_trades']:>15d}")

        if bal["positions"]:
            print(f"\n  {'Ticker':<8} {'Qty':>5} {'Entry':>8} {'Now':>8} {'P&L':>10} {'%':>7}")
            print(f"  {'-'*48}")
            for p in bal["positions"]:
                print(f"  {p['ticker']:<8} {p['qty']:>5} ${p['entry_price']:>7.2f} "
                      f"${p['current_price']:>7.2f} {p['pnl_krw']:>+10,.0f} {p['pnl_pct']:>+6.2%}")

        if self.trade_history:
            print(f"\n  Recent Trades:")
            for t in self.trade_history[-5:]:
                pnl = f" P&L:{t.get('pnl_pct',0):+.2%}" if t["side"] == "SELL" else ""
                print(f"    {t['time']} {t['side']:4s} {t['ticker']:6s} "
                      f"{t['qty']}주 @ ${t['price_usd']:.2f}{pnl}")
        print("=" * 55)

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash_krw": self.cash_krw,
            "positions": self.positions,
            "trade_history": self.trade_history,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self.save_path, "r") as f:
            data = json.load(f)
        self.cash_krw = data["cash_krw"]
        self.positions = data["positions"]
        self.trade_history = data.get("trade_history", [])

    def reset(self, capital: float = 100_000_000) -> None:
        """Reset account to initial state."""
        self.cash_krw = capital
        self.positions = []
        self.trade_history = []
        self._save()
        logger.info(f"Paper account reset: {capital:,.0f} KRW")
