"""V3.3 BacktestEngine integration tests (F3 fix).

Verifies V3.3 ctx.actions consumption path:
  - features.exit_thesis OFF → V3.2.1 path (parity)
  - features.exit_thesis ON  → V3.3 path (BookOptimizer.ctx.actions dispatch)

Helper tests:
  - _convert_to_position_states (dict → PositionState)
  - _check_triggers_v33 (ExitRules → trigger map)
  - _handle_exit_action / _handle_trim_action
  - _handle_add_new_action / _handle_pyramid_action
  - _process_book_actions (multi-action dispatch)

Synthesis-only — no actual backtest run (uses BacktestEngine helpers
directly with synthesized inputs).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from v3.backtest.engine import BacktestEngine, TradeRecord
from v3.config.schema import load_config
from v3.strategy.types import BookAction, DecisionContext, PositionState


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def engine(tmp_path):
    """Real BacktestEngine instance (lazy init dependencies)."""
    cfg = load_config()
    # Patch heavy deps to avoid file I/O
    with patch("v3.backtest.engine.MacroCollector"), \
         patch("v3.backtest.engine.RegimeDetectorV2"):
        eng = BacktestEngine(cfg)
    return eng


def _position(
    ticker: str = "AAPL",
    entry_price: float = 200.0,
    weight: float = 0.20,
    hold_days: int = 3,
    capital_allocated: float = 20_000_000,
) -> dict:
    return {
        "ticker": ticker,
        "entry_date": "2026-05-01",
        "entry_price": entry_price,
        "weight": weight,
        "hold_days": hold_days,
        "entry_vol": 0.20,
        "vol_score": 0.5,
        "opportunity": 0.005,
        "capital_allocated": capital_allocated,
        "last_unrealized": 0.0,
    }


def _today_data(
    ticker: str = "AAPL",
    close: float = 210.0,
    open_: float = 209.0,
    low: float = 208.0,
    vol: float = 0.20,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-05-09"),
        "ticker": ticker,
        "open": open_, "high": close * 1.005, "low": low,
        "close": close, "volume": 1_000_000,
        "vol_cc_5d": vol,
    }])


def _action(
    action_type: str,
    ticker: str = "AAPL",
    target_weight: float = 0.0,
    current_weight: float = 0.20,
    reason: str = "test",
    expected_impact: float | None = 0.005,
    replacement_ticker: str | None = None,
) -> BookAction:
    return BookAction(
        date=pd.Timestamp("2026-05-09"),
        action_type=action_type,
        ticker=ticker,
        target_weight=target_weight,
        current_weight=current_weight,
        reason=reason,
        source_policy="test",
        expected_impact=expected_impact,
        replacement_ticker=replacement_ticker,
    )


# ──────────────────────────────────────────────────────────────
# _convert_to_position_states
# ──────────────────────────────────────────────────────────────
class TestConvertToPositionStates:
    def test_basic_conversion(self, engine):
        positions = [_position(ticker="AAPL", entry_price=200.0)]
        today = _today_data(ticker="AAPL", close=210.0)
        states = engine._convert_to_position_states(positions, today)
        assert len(states) == 1
        s = states[0]
        assert isinstance(s, PositionState)
        assert s.ticker == "AAPL"
        assert s.current_price == 210.0
        # pnl = 210/200 - 1 = 0.05
        assert s.pnl_pct == pytest.approx(0.05)
        # holding_days = pos["hold_days"] + 1 = 4
        assert s.holding_days == 4

    def test_missing_today_data_uses_entry_price(self, engine):
        positions = [_position(ticker="AAPL", entry_price=200.0)]
        # today_data has different ticker
        today = _today_data(ticker="MSFT")
        states = engine._convert_to_position_states(positions, today)
        # Falls back to entry_price → pnl_pct = 0
        assert states[0].current_price == 200.0
        assert states[0].pnl_pct == 0.0

    def test_thesis_alive_when_profitable(self, engine):
        positions = [_position(entry_price=200.0)]
        today = _today_data(close=210.0)
        states = engine._convert_to_position_states(positions, today)
        assert states[0].thesis_alive is True

    def test_thesis_alive_false_when_loss(self, engine):
        positions = [_position(entry_price=200.0)]
        today = _today_data(close=190.0)
        states = engine._convert_to_position_states(positions, today)
        assert states[0].thesis_alive is False


# ──────────────────────────────────────────────────────────────
# _check_triggers_v33
# ──────────────────────────────────────────────────────────────
class TestCheckTriggersV33:
    def test_no_exit_returns_max_hold(self, engine):
        # Profit not yet at TP target, low vol stable → no exit
        positions = [_position(entry_price=200.0)]
        today = _today_data(close=201.0, vol=0.20)
        triggers = engine._check_triggers_v33(positions, today, portfolio_deployed=0.5)
        assert triggers["AAPL"] == "max_hold"  # default no-exit trigger

    def test_returns_exit_trigger_when_should_exit(self, engine):
        # large profit → profit_take trigger
        positions = [_position(entry_price=200.0, hold_days=0)]  # Day 1 → +5% TP
        today = _today_data(close=212.0)  # +6% > Day 1 TP 5%
        triggers = engine._check_triggers_v33(positions, today, portfolio_deployed=0.5)
        # ExitRules returns "profit_take" or similar
        assert triggers["AAPL"] != "max_hold"  # some specific trigger

    def test_missing_today_skipped(self, engine):
        positions = [_position(ticker="AAPL"), _position(ticker="MSFT")]
        today = _today_data(ticker="AAPL")  # only AAPL
        triggers = engine._check_triggers_v33(positions, today, portfolio_deployed=0.5)
        assert "AAPL" in triggers
        assert "MSFT" not in triggers


# ──────────────────────────────────────────────────────────────
# _handle_exit_action
# ──────────────────────────────────────────────────────────────
class TestHandleExitAction:
    def test_exit_creates_trade_and_removes_position(self, engine):
        positions = [_position(ticker="AAPL", entry_price=200.0,
                               capital_allocated=20_000_000)]
        today = _today_data(ticker="AAPL", close=210.0)
        action = _action("EXIT", ticker="AAPL", reason="profit_take")

        new_pos, trade, cash_delta, pnl_delta = engine._handle_exit_action(
            action, positions, today, "2026-05-09",
        )
        assert len(new_pos) == 0  # position removed
        assert isinstance(trade, TradeRecord)
        assert trade.ticker == "AAPL"
        assert trade.exit_reason == "profit_take"
        # gross_ret ≈ 210*(1-slippage)/200 - 1 ≈ +5% minus slip
        assert trade.gross_return > 0
        # cash recovered (positive delta)
        assert cash_delta > 0

    def test_exit_unknown_ticker_no_op(self, engine):
        positions = [_position(ticker="AAPL")]
        today = _today_data(ticker="AAPL")
        action = _action("EXIT", ticker="UNKNOWN")
        new_pos, trade, cash_delta, pnl_delta = engine._handle_exit_action(
            action, positions, today, "2026-05-09",
        )
        assert new_pos == positions
        assert trade is None
        assert cash_delta == 0.0


# ──────────────────────────────────────────────────────────────
# _handle_trim_action
# ──────────────────────────────────────────────────────────────
class TestHandleTrimAction:
    def test_trim_creates_partial_trade(self, engine):
        positions = [_position(ticker="AAPL", entry_price=200.0,
                               weight=0.30, capital_allocated=30_000_000)]
        today = _today_data(ticker="AAPL", close=210.0)
        # Trim from 0.30 → 0.21 (30% reduction)
        action = _action("TRIM", ticker="AAPL",
                         target_weight=0.21, current_weight=0.30,
                         reason="weak_residual_edge")
        new_pos, trade, cash_delta, pnl_delta = engine._handle_trim_action(
            action, positions, today, "2026-05-09",
        )
        assert len(new_pos) == 1  # position kept
        assert new_pos[0]["weight"] == pytest.approx(0.21)
        assert new_pos[0]["capital_allocated"] == pytest.approx(
            30_000_000 * 0.7, rel=1e-3
        )
        assert isinstance(trade, TradeRecord)
        assert "TRIM:" in trade.exit_reason
        assert cash_delta > 0  # partial recovery

    def test_trim_to_full_weight_no_op(self, engine):
        positions = [_position(weight=0.20)]
        today = _today_data()
        # target == current → no trim
        action = _action("TRIM", target_weight=0.20, current_weight=0.20)
        new_pos, trade, cash_delta, _ = engine._handle_trim_action(
            action, positions, today, "2026-05-09",
        )
        assert trade is None
        assert cash_delta == 0.0


# ──────────────────────────────────────────────────────────────
# _handle_add_new_action
# ──────────────────────────────────────────────────────────────
class TestHandleAddNewAction:
    def test_add_new_creates_position(self, engine):
        positions = []
        today = _today_data(ticker="AAPL", open_=200.0, close=205.0)
        vol_scores = pd.DataFrame([{
            "ticker": "AAPL", "vol_score": 0.7, "confidence": 0.8,
        }])
        action = _action("ADD_NEW", ticker="AAPL",
                         target_weight=0.20, current_weight=0.0)
        cash = 100_000_000
        monthly = set()

        new_pos, cash_delta = engine._handle_add_new_action(
            action, positions, today, vol_scores,
            "2026-05-09", cash, monthly,
        )
        assert len(new_pos) == 1
        assert new_pos[0]["ticker"] == "AAPL"
        assert new_pos[0]["weight"] == 0.20
        assert new_pos[0]["vol_score"] == 0.7
        # Cash delta negative (capital reserved)
        assert cash_delta < 0
        # ~20M allocated
        assert cash_delta == pytest.approx(-20_000_000, rel=0.01)
        assert "AAPL" in monthly

    def test_add_new_skips_already_held(self, engine):
        positions = [_position(ticker="AAPL")]
        today = _today_data(ticker="AAPL")
        vol_scores = pd.DataFrame()
        action = _action("ADD_NEW", ticker="AAPL")
        new_pos, cash_delta = engine._handle_add_new_action(
            action, positions, today, vol_scores, "2026-05-09", 100_000_000, set(),
        )
        assert new_pos == positions  # unchanged
        assert cash_delta == 0.0

    def test_add_new_below_min_order_skipped(self, engine):
        positions = []
        today = _today_data(ticker="AAPL")
        vol_scores = pd.DataFrame()
        # Tiny weight → allocated < min_order_amount_krw (5M)
        action = _action("ADD_NEW", ticker="AAPL",
                         target_weight=0.01, current_weight=0.0)
        new_pos, cash_delta = engine._handle_add_new_action(
            action, positions, today, vol_scores, "2026-05-09", 100_000_000, set(),
        )
        assert new_pos == positions
        assert cash_delta == 0.0


# ──────────────────────────────────────────────────────────────
# _handle_pyramid_action
# ──────────────────────────────────────────────────────────────
class TestHandlePyramidAction:
    def test_pyramid_increases_weight_and_avg_entry(self, engine):
        positions = [_position(
            ticker="AAPL", entry_price=200.0, weight=0.20,
            capital_allocated=20_000_000,
        )]
        today = _today_data(ticker="AAPL", open_=210.0, close=212.0)
        # ADD_TO_WINNER from 0.20 → 0.30
        action = _action("ADD_TO_WINNER", ticker="AAPL",
                         target_weight=0.30, current_weight=0.20)
        cash = 100_000_000
        new_pos, cash_delta = engine._handle_pyramid_action(
            action, positions, today, "2026-05-09", cash,
        )
        assert len(new_pos) == 1
        assert new_pos[0]["weight"] == 0.30
        # Weighted-avg entry > 200 (added at 210+slippage)
        assert new_pos[0]["entry_price"] > 200.0
        # Cash delta negative
        assert cash_delta < 0

    def test_pyramid_unknown_ticker_no_op(self, engine):
        positions = [_position(ticker="AAPL")]
        today = _today_data(ticker="AAPL")
        action = _action("ADD_TO_WINNER", ticker="UNKNOWN",
                         target_weight=0.30, current_weight=0.0)
        new_pos, cash_delta = engine._handle_pyramid_action(
            action, positions, today, "2026-05-09", 100_000_000,
        )
        assert new_pos == positions
        assert cash_delta == 0.0

    def test_pyramid_target_below_current_no_op(self, engine):
        positions = [_position(weight=0.30)]
        today = _today_data()
        action = _action("ADD_TO_WINNER", target_weight=0.20, current_weight=0.30)
        new_pos, cash_delta = engine._handle_pyramid_action(
            action, positions, today, "2026-05-09", 100_000_000,
        )
        assert cash_delta == 0.0


# ──────────────────────────────────────────────────────────────
# _process_book_actions — multi-action dispatch
# ──────────────────────────────────────────────────────────────
class TestProcessBookActions:
    def _ctx(self, actions: list[BookAction]) -> DecisionContext:
        return DecisionContext(actions=actions, signal=None, candidates=[])

    def test_keep_no_op_advances_hold_days(self, engine):
        positions = [_position(ticker="AAPL", hold_days=3)]
        today = _today_data(ticker="AAPL", close=210.0)
        vol_scores = pd.DataFrame()
        ctx = self._ctx([_action("KEEP", ticker="AAPL")])
        new_pos, trades, cash, pnl = engine._process_book_actions(
            ctx, positions, today, vol_scores,
            "2026-05-09", 100_000_000, set(),
        )
        assert len(new_pos) == 1
        assert len(trades) == 0
        # hold_days incremented
        assert new_pos[0]["hold_days"] == 4

    def test_exit_then_add_new(self, engine):
        positions = [_position(ticker="AAPL")]
        today = pd.concat([
            _today_data(ticker="AAPL", close=210.0),
            _today_data(ticker="MSFT", close=305.0, open_=300.0),
        ])
        vol_scores = pd.DataFrame([
            {"ticker": "MSFT", "vol_score": 0.6, "confidence": 0.7},
        ])
        ctx = self._ctx([
            _action("EXIT", ticker="AAPL", reason="profit_take"),
            _action("ADD_NEW", ticker="MSFT",
                    target_weight=0.15, current_weight=0.0),
        ])
        new_pos, trades, cash, pnl = engine._process_book_actions(
            ctx, positions, today, vol_scores,
            "2026-05-09", 100_000_000, set(),
        )
        # AAPL exited, MSFT added
        tickers = {p["ticker"] for p in new_pos}
        assert tickers == {"MSFT"}
        assert len(trades) == 1
        assert trades[0].ticker == "AAPL"

    def test_no_action_ignored(self, engine):
        positions = []
        today = _today_data()
        ctx = self._ctx([_action("NO_ACTION", ticker="")])
        new_pos, trades, cash, pnl = engine._process_book_actions(
            ctx, positions, today, pd.DataFrame(),
            "2026-05-09", 100_000_000, set(),
        )
        assert new_pos == []
        assert trades == []

    def test_blocked_ignored(self, engine):
        positions = [_position(ticker="AAPL")]
        today = _today_data(ticker="AAPL")
        ctx = self._ctx([_action("BLOCKED", ticker="AAPL")])
        new_pos, trades, _, _ = engine._process_book_actions(
            ctx, positions, today, pd.DataFrame(),
            "2026-05-09", 100_000_000, set(),
        )
        assert len(new_pos) == 1  # position unchanged
        assert len(trades) == 0
