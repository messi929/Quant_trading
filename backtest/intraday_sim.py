"""Intraday exit simulation from daily OHLC data.

Simulates profit-taking, stop-loss, and time-based exits using
daily high/low/close prices. Conservative assumption: stop-loss
is checked before profit-taking (worst-case scenario).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitResult:
    """Result of intraday exit simulation."""
    exit_price: float
    exit_reason: str  # "profit_take", "stop_loss", "time_exit", "session_close", "hold"
    return_pct: float
    exited: bool


def simulate_intraday_exit(
    entry_price: float,
    high: float,
    low: float,
    close: float,
    profit_take_pct: float = 0.025,
    stop_loss_pct: float = 0.02,
    stop_first: bool = True,
) -> ExitResult:
    """Simulate intraday exit from daily OHLC.

    Args:
        entry_price: Entry price.
        high: Day's high price.
        low: Day's low price.
        close: Day's closing price.
        profit_take_pct: Profit target (e.g., 0.025 = +2.5%).
        stop_loss_pct: Stop loss level (e.g., 0.02 = -2%).
        stop_first: If True, check stop-loss before profit-taking
                    (conservative, default). This matters when both
                    high and low trigger within the same day.

    Returns:
        ExitResult with exit_price, reason, return, and whether exited.
    """
    profit_target = entry_price * (1 + profit_take_pct)
    stop_target = entry_price * (1 - stop_loss_pct)

    hit_profit = high >= profit_target
    hit_stop = low <= stop_target

    if hit_stop and hit_profit:
        # Both triggered in same day — conservative: assume worst case
        if stop_first:
            exit_price = stop_target
            reason = "stop_loss"
        else:
            exit_price = profit_target
            reason = "profit_take"
    elif hit_stop:
        exit_price = stop_target
        reason = "stop_loss"
    elif hit_profit:
        exit_price = profit_target
        reason = "profit_take"
    else:
        # Neither triggered — hold or close at end of day
        return ExitResult(
            exit_price=close,
            exit_reason="hold",
            return_pct=(close - entry_price) / entry_price,
            exited=False,
        )

    return_pct = (exit_price - entry_price) / entry_price
    return ExitResult(
        exit_price=exit_price,
        exit_reason=reason,
        return_pct=return_pct,
        exited=True,
    )


def simulate_multi_day_exit(
    entry_price: float,
    daily_ohlc: list[dict],
    profit_take_pct: float = 0.025,
    profit_take_full_pct: float = 0.05,
    stop_loss_pct: float = 0.02,
    max_hold_days: int = 3,
    day_trade_default: bool = True,
    stop_first: bool = True,
) -> ExitResult:
    """Simulate exit over multiple days.

    Args:
        entry_price: Entry price.
        daily_ohlc: List of {"high", "low", "close"} dicts, one per day.
        profit_take_pct: Partial profit target.
        profit_take_full_pct: Full exit profit target.
        stop_loss_pct: Stop loss level.
        max_hold_days: Maximum holding period.
        day_trade_default: If True, exit at close of day 1.
        stop_first: Conservative stop-first assumption.

    Returns:
        ExitResult for the final exit.
    """
    if not daily_ohlc:
        return ExitResult(entry_price, "no_data", 0.0, False)

    for day_idx, ohlc in enumerate(daily_ohlc):
        high = ohlc["high"]
        low = ohlc["low"]
        close = ohlc["close"]

        result = simulate_intraday_exit(
            entry_price, high, low, close,
            profit_take_pct=profit_take_full_pct,  # Use full exit target
            stop_loss_pct=stop_loss_pct,
            stop_first=stop_first,
        )

        # Profit or stop triggered
        if result.exited:
            return result

        # Day trade default: exit at close of day 1
        if day_trade_default and day_idx == 0:
            return ExitResult(
                exit_price=close,
                exit_reason="session_close",
                return_pct=(close - entry_price) / entry_price,
                exited=True,
            )

        # Max hold days reached
        if day_idx + 1 >= max_hold_days:
            return ExitResult(
                exit_price=close,
                exit_reason="max_hold",
                return_pct=(close - entry_price) / entry_price,
                exited=True,
            )

    # Fell through (shouldn't happen with max_hold_days)
    last_close = daily_ohlc[-1]["close"]
    return ExitResult(
        exit_price=last_close,
        exit_reason="session_close",
        return_pct=(last_close - entry_price) / entry_price,
        exited=True,
    )
