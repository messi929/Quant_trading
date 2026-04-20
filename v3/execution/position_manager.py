"""Position persistence and ghost removal.

Tracks open positions, persists across restarts,
and handles ghost position cleanup.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


class PositionManager:
    """Manages open positions with disk persistence."""

    GHOST_MAX_RETRIES = 3
    COOLDOWN_DAYS = 5
    COOLDOWN_LOSS_COUNT = 2
    MAX_CONSECUTIVE_ENTRY = 3

    def __init__(self, save_dir: str = "v3/saved_models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.positions_path = self.save_dir / "open_positions.json"
        self.cooldown_path = self.save_dir / "ticker_cooldown.json"
        self.history_path = self.save_dir / "entry_history.json"
        self.positions: list[dict] = []
        self.cooldown: dict[str, list] = {}
        self.sell_retries: dict[str, int] = {}
        self.entry_history: dict[str, list[str]] = {}  # ticker → [dates]
        self._load()

    def _load(self) -> None:
        if self.positions_path.exists():
            with open(self.positions_path, "r") as f:
                self.positions = json.load(f)
            logger.info(f"Loaded {len(self.positions)} positions")
        if self.cooldown_path.exists():
            with open(self.cooldown_path, "r") as f:
                self.cooldown = json.load(f)
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                data = json.load(f)
                self.entry_history = data.get("entry_history", {})
                self.sell_retries = data.get("sell_retries", {})
            logger.info(
                f"Loaded entry_history: {len(self.entry_history)} tickers, "
                f"sell_retries: {len(self.sell_retries)}"
            )

    def save(self) -> None:
        with open(self.positions_path, "w") as f:
            json.dump(self.positions, f, indent=2, default=str)
        with open(self.cooldown_path, "w") as f:
            json.dump(self.cooldown, f, indent=2, default=str)
        with open(self.history_path, "w") as f:
            json.dump(
                {"entry_history": self.entry_history, "sell_retries": self.sell_retries},
                f, indent=2, default=str,
            )

    def add_position(self, position: dict) -> None:
        self.positions.append(position)
        ticker = position["ticker"]
        today = position.get("entry_date", datetime.now().strftime("%Y-%m-%d"))
        if ticker not in self.entry_history:
            self.entry_history[ticker] = []
        self.entry_history[ticker].append(today)
        self.save()
        logger.info(f"Position added: {ticker} @ {position.get('entry_price', '?')}")

    def remove_position(self, ticker: str) -> dict | None:
        for i, p in enumerate(self.positions):
            if p["ticker"] == ticker:
                removed = self.positions.pop(i)
                self.sell_retries.pop(ticker, None)
                self.save()
                logger.info(f"Position removed: {ticker}")
                return removed
        return None

    def get_position(self, ticker: str) -> dict | None:
        for p in self.positions:
            if p["ticker"] == ticker:
                return p
        return None

    def get_open_tickers(self) -> list[str]:
        return [p["ticker"] for p in self.positions]

    def count(self) -> int:
        return len(self.positions)

    # ── Ghost Position Handling ──────────────────────────────

    def record_sell_failure(self, ticker: str) -> bool:
        """Record a sell failure. Returns True if position should be force-removed."""
        self.sell_retries[ticker] = self.sell_retries.get(ticker, 0) + 1
        if self.sell_retries[ticker] >= self.GHOST_MAX_RETRIES:
            logger.warning(f"Ghost position: {ticker} — {self.GHOST_MAX_RETRIES} sell failures, force removing")
            self.remove_position(ticker)
            # Clear entry_history to prevent consecutive-entry veto from a ghost
            self.entry_history.pop(ticker, None)
            self.save()
            return True
        return False

    # ── Cooldown ─────────────────────────────────────────────

    def record_loss(self, ticker: str, date: str) -> None:
        if ticker not in self.cooldown:
            self.cooldown[ticker] = []
        self.cooldown[ticker].append(date)
        # Keep only recent entries
        self.cooldown[ticker] = self.cooldown[ticker][-10:]
        self.save()

    def is_cooled_down(self, ticker: str, current_date: str) -> bool:
        """Returns True if ticker is in cooldown (too many recent losses)."""
        if ticker not in self.cooldown:
            return False

        cutoff = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=self.COOLDOWN_DAYS)).strftime("%Y-%m-%d")
        recent_losses = [d for d in self.cooldown[ticker] if d >= cutoff]

        if len(recent_losses) >= self.COOLDOWN_LOSS_COUNT:
            logger.info(f"Cooldown active: {ticker} ({len(recent_losses)} losses in {self.COOLDOWN_DAYS}d)")
            return True
        return False

    def is_consecutive_entry(self, ticker: str, current_date: str) -> bool:
        """Returns True if ticker has been entered too many consecutive days."""
        if ticker not in self.entry_history:
            return False

        recent = self.entry_history[ticker][-self.MAX_CONSECUTIVE_ENTRY:]
        if len(recent) < self.MAX_CONSECUTIVE_ENTRY:
            return False

        # Check if last N entries are consecutive days
        try:
            entry_dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in recent])
            span = (entry_dates[-1] - entry_dates[0]).days
            if span <= self.MAX_CONSECUTIVE_ENTRY + 1:
                logger.info(f"Consecutive entry limit: {ticker} ({len(recent)} in {span}d)")
                return True
        except ValueError:
            pass
        return False

    # ── Monthly Trade Count ──────────────────────────────────

    def monthly_trade_count(self, current_date: str) -> int:
        """Count trades this month across all tickers."""
        month = current_date[:7]
        count = 0
        for ticker, dates in self.entry_history.items():
            count += sum(1 for d in dates if d.startswith(month))
        return count
