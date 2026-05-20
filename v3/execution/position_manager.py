"""Position persistence and ghost removal.

Tracks open positions, persists across restarts,
and handles ghost position cleanup.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger


class PositionManager:
    """Manages open positions with disk persistence."""

    GHOST_MAX_RETRIES = 3
    COOLDOWN_DAYS = 5            # TRADING days (weekends/holidays excluded)
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
        """Returns True if ticker is in cooldown (too many recent losses).

        Window is counted in TRADING days, not calendar days. A calendar-day
        window leaked across weekends (2026-05-20 bug): MU lost Thu 5/14 +
        Fri 5/15 but on Wed 5/20 the 5/14 loss was 6 calendar days back → out
        of the 5-day window → re-entry allowed. np.busday_count excludes
        weekends (holidays not modeled — acceptable for this guard rail).
        """
        if ticker not in self.cooldown:
            return False

        cur = datetime.strptime(current_date, "%Y-%m-%d").date()
        recent_losses = []
        for d in self.cooldown[ticker]:
            try:
                loss = datetime.strptime(d, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                logger.warning(f"Skipping malformed cooldown date for {ticker}: {d!r}")
                continue
            # Trading days in [loss, cur). 0 ≤ n < COOLDOWN_DAYS → within window.
            trading_days = int(np.busday_count(loss, cur))
            if 0 <= trading_days < self.COOLDOWN_DAYS:
                recent_losses.append(d)

        if len(recent_losses) >= self.COOLDOWN_LOSS_COUNT:
            logger.info(
                f"Cooldown active: {ticker} ({len(recent_losses)} losses in "
                f"{self.COOLDOWN_DAYS} trading days)"
            )
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
        """Count UNIQUE tickers entered this month (Phase 25.1 옵션 C).

        CLAUDE.md "Monthly Trade Cap 재설계" 적용. 4/11~4/24 데이터에서 BUY
        7회 중 5회가 FANG churn으로 4월 한도(7회)를 조기 소진했고, 이후
        4/27~4/30 8세션 동안 31개 후보가 monthly_trades에 막혀 QQQ +1.55%
        구간을 통째로 놓쳤음. 페르소나 원칙 1 ("확신 있을 때만")의 진짜
        결정자가 conviction이 아니라 임의 cap이 된 implementation 결함을
        해소하기 위해 "건수"가 아닌 "unique 종목 수"로 카운트한다. FANG 5회
        churn = 1로 카운트.
        """
        month = current_date[:7]
        unique_tickers = {
            ticker for ticker, dates in self.entry_history.items()
            if any(d.startswith(month) for d in dates)
        }
        return len(unique_tickers)
