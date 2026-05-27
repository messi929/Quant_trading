"""Exit rules — time-decay TP, dynamic stops, vol contraction quality.

Medallion-inspired enhancements:
  1. Time-decay profit target (edge erodes daily; tighten TP over hold period)
  2. Adaptive TP by confidence (high conf → tighter, low conf → wider)
  3. MAE-based dynamic stop (trailing stop tightens after adverse move)
  4. Vol contraction persistence check (require 3-day sustained contraction)
  5. Portfolio-proportional daily limit
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    exit_price: float
    return_pct: float
    max_adverse_excursion: float = 0.0
    trailing_stop_triggered: bool = False


class ExitRules:
    """Determines when to exit a position with time-decay and dynamic stops."""

    # Profit target decay: edge erodes over holding period
    TIME_DECAY_TARGETS = {
        0: 0.050,   # Day 1: 5.0%
        1: 0.040,   # Day 2: 4.0%
        2: 0.030,   # Day 3: 3.0%
        3: 0.020,   # Day 4: 2.0%
        4: 0.015,   # Day 5: 1.5%
    }

    def __init__(
        self,
        profit_take_pct: float = 0.05,
        max_hold_days: int = 5,
        vol_contraction_ratio: float = 0.70,
        daily_loss_limit: float = 0.015,
        use_time_decay_tp: bool = True,
        mae_stop_threshold: float = -0.03,   # MAE -3% → tighten stop
        mae_stop_tightened: float = -0.025,   # After MAE trigger → -2.5% stop
        trailing_activation: float = 0.03,   # Peak ≥ +3% → activate trailing
        trailing_drop: float = 0.02,         # Drop ≥ 2% from peak → exit
        use_trailing_stop: bool = True,      # A1 (2026-05-27): trailing stop 도입
    ):
        self.profit_take_pct = profit_take_pct
        self.max_hold_days = max_hold_days
        self.vol_contraction_ratio = vol_contraction_ratio
        self.daily_loss_limit = daily_loss_limit
        self.use_time_decay = use_time_decay_tp
        self.mae_threshold = mae_stop_threshold
        self.mae_tightened = mae_stop_tightened
        self.trailing_activation = trailing_activation
        self.trailing_drop = trailing_drop
        self.use_trailing_stop = use_trailing_stop

    def check(
        self,
        entry_price: float,
        current_price: float,
        hold_days: int,
        entry_vol: float,
        current_vol: float,
        portfolio_daily_pnl: float = 0.0,
        confidence: float = 0.5,
        low_price: float = 0.0,
        portfolio_deployed: float = 0.5,
        vol_history: list[float] | None = None,
        peak_price: float = 0.0,
    ) -> ExitDecision:
        """Check all exit conditions with Medallion-grade enhancements.

        peak_price: 진입 후 관측된 최고가 (intraday high 포함). A1 trailing stop
        용. 0이면 trailing 비활성 (backward-compat).
        """
        return_pct = (current_price / entry_price - 1) if entry_price > 0 else 0
        mae = (low_price / entry_price - 1) if (entry_price > 0 and low_price > 0) else 0
        peak_return = (peak_price / entry_price - 1) if (entry_price > 0 and peak_price > 0) else return_pct

        # P1: Portfolio-proportional daily stop
        dynamic_limit = self._dynamic_daily_limit(portfolio_deployed)
        if portfolio_daily_pnl <= -dynamic_limit:
            return ExitDecision(True, "portfolio_stop", current_price, return_pct,
                                max_adverse_excursion=mae)

        # P2: MAE-based dynamic stop (trailing tightens after adverse move)
        # Trigger if EITHER: (a) intraday MAE hit threshold, OR (b) close below tightened stop
        if mae < self.mae_threshold or return_pct < self.mae_tightened:
            if return_pct < self.mae_tightened:
                return ExitDecision(True, "dynamic_stop_mae", current_price, return_pct,
                                    max_adverse_excursion=mae, trailing_stop_triggered=True)

        # P2.5: Trailing stop (A1, 2026-05-27)
        # peak ≥ +3% 도달 후, peak 대비 2% drop 시 청산 — winner 보호 + 비대칭
        # 손익비 개선. profit_take time-decay 이전에 평가하여 큰 winner riding.
        if self.use_trailing_stop and peak_return >= self.trailing_activation:
            drawdown_from_peak = peak_return - return_pct
            if drawdown_from_peak >= self.trailing_drop:
                return ExitDecision(True, "trailing_stop", current_price, return_pct,
                                    max_adverse_excursion=mae, trailing_stop_triggered=True)

        # P3: Time-decay profit take (confidence-adjusted)
        tp_target = self._get_profit_target(hold_days, confidence, entry_vol, current_vol)
        if return_pct >= tp_target:
            return ExitDecision(True, "profit_take", current_price, return_pct,
                                max_adverse_excursion=mae)

        # P4: Max hold days
        if hold_days >= self.max_hold_days:
            return ExitDecision(True, "max_hold", current_price, return_pct,
                                max_adverse_excursion=mae)

        # P5: Vol contraction (sustained, not noise)
        if entry_vol > 0 and current_vol > 0:
            if self._check_vol_contraction(entry_vol, current_vol, vol_history):
                return ExitDecision(True, "vol_contraction", current_price, return_pct,
                                    max_adverse_excursion=mae)

        return ExitDecision(False, "hold", current_price, return_pct,
                            max_adverse_excursion=mae)

    def _get_profit_target(
        self, hold_days: int, confidence: float, entry_vol: float, current_vol: float,
    ) -> float:
        """Time-decay + confidence-adaptive profit target."""
        if self.use_time_decay:
            base = self.TIME_DECAY_TARGETS.get(hold_days, 0.01)
        else:
            base = self.profit_take_pct

        # Confidence adjustment: high conf → tighter (take faster), low → wider
        if confidence > 0.75:
            base *= 0.7
        elif confidence < 0.45:
            base *= 1.3

        # Vol regime: high current vol → take smaller profits (more uncertain)
        if entry_vol > 0 and current_vol > 0:
            vol_ratio = current_vol / entry_vol
            if vol_ratio > 1.2:   # Vol still expanding
                base *= 0.85      # Take profits sooner
            elif vol_ratio < 0.8: # Vol contracting
                base *= 1.1       # Can wait a bit longer

        return max(base, 0.005)   # Minimum 0.5% target

    def _check_vol_contraction(
        self, entry_vol: float, current_vol: float, vol_history: list[float] | None,
    ) -> bool:
        """Vol contraction check with persistence requirement.

        Require 2+ consecutive days of declining vol to confirm contraction.
        Don't exit on one-day vol noise.
        """
        vol_ratio = current_vol / entry_vol
        if vol_ratio > self.vol_contraction_ratio:
            return False   # Not contracted enough

        # If no history, use simple ratio check
        if not vol_history or len(vol_history) < 3:
            return vol_ratio <= self.vol_contraction_ratio

        # Persistence: last 2 days must both be declining
        recent = vol_history[-3:]
        sustained = (recent[-1] < recent[-2]) and (recent[-2] < recent[-3])

        return sustained

    def _dynamic_daily_limit(self, portfolio_deployed: float) -> float:
        """Tighter stop when more capital is deployed."""
        if portfolio_deployed > 0.90:
            return 0.010
        elif portfolio_deployed > 0.70:
            return 0.012
        elif portfolio_deployed < 0.30:
            return 0.020
        return self.daily_loss_limit

    def simulate_multi_day(
        self,
        entry_price: float,
        daily_ohlc: list[dict],
        entry_vol: float,
        daily_vols: list[float] | None = None,
        confidence: float = 0.5,
    ) -> ExitDecision:
        """Simulate exit over multiple days with all enhanced rules."""
        if not daily_ohlc:
            return ExitDecision(False, "hold", entry_price, 0.0)

        worst_mae = 0.0

        for day_idx, ohlc in enumerate(daily_ohlc):
            close = ohlc["close"]
            high = ohlc["high"]
            low = ohlc["low"]

            current_vol = daily_vols[day_idx] if daily_vols and day_idx < len(daily_vols) else entry_vol
            vol_hist = daily_vols[:day_idx + 1] if daily_vols else None

            # Track MAE
            low_return = (low / entry_price - 1) if entry_price > 0 else 0
            worst_mae = min(worst_mae, low_return)

            # Time-decay profit take using intraday high
            tp_target = self._get_profit_target(day_idx, confidence, entry_vol, current_vol)
            high_return = (high / entry_price - 1) if entry_price > 0 else 0
            if high_return >= tp_target:
                take_price = entry_price * (1 + tp_target)
                return ExitDecision(True, "profit_take", take_price, tp_target,
                                    max_adverse_excursion=worst_mae)

            # MAE dynamic stop
            if worst_mae < self.mae_threshold:
                close_return = (close / entry_price - 1)
                if close_return < self.mae_tightened:
                    return ExitDecision(True, "dynamic_stop_mae", close, close_return,
                                        max_adverse_excursion=worst_mae, trailing_stop_triggered=True)

            # Vol contraction (sustained)
            if self._check_vol_contraction(entry_vol, current_vol, vol_hist):
                return_pct = (close / entry_price - 1)
                return ExitDecision(True, "vol_contraction", close, return_pct,
                                    max_adverse_excursion=worst_mae)

            # Max hold
            if day_idx + 1 >= self.max_hold_days:
                return_pct = (close / entry_price - 1)
                return ExitDecision(True, "max_hold", close, return_pct,
                                    max_adverse_excursion=worst_mae)

        last_close = daily_ohlc[-1]["close"]
        return ExitDecision(False, "hold", last_close,
                            (last_close / entry_price - 1),
                            max_adverse_excursion=worst_mae)
