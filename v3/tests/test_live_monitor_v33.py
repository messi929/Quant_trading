"""LivePipeline.monitor() V3.3 routing tests.

Verifies:
  - monitor() routes to V3.2.1 path when features.exit_thesis OFF
  - monitor() routes to V3.2.1 path when _last_ctx is None
  - monitor() routes to _monitor_v33 when both conditions met
  - _monitor_v33 skips positions with no ExitRules trigger
  - _monitor_v33 calls ExitThesisEngine.decide() when trigger fires
  - _monitor_v33 routes resulting BookActions through execute_actions()
  - _monitor_v33 updates pyramid/rotation cross-session state
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from v3.config.schema import V3Config
from v3.pipeline.live_pipeline import LivePipeline
from v3.strategy.exit_thesis import ExitPolicy, ExitThesisEngine
from v3.strategy.types import (
    BookAction,
    DecisionContext,
    EdgeCandidate,
    PositionState,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def cfg_features_off():
    cfg = V3Config()
    cfg.features.exit_thesis = False
    return cfg


@pytest.fixture
def cfg_features_on():
    cfg = V3Config()
    cfg.features.exit_thesis = True
    return cfg


def _make_pipeline(cfg, ctx, executor, exit_engine, age_hours: float = 1.0,
                   state_path=None):
    """Construct a LivePipeline-like instance without running heavy __init__."""
    import tempfile
    pipe = LivePipeline.__new__(LivePipeline)
    pipe.cfg = cfg
    pipe._last_ctx = ctx
    pipe._opportunity_at = (
        pd.Timestamp.now() - pd.Timedelta(hours=age_hours)
        if age_hours is not None else None
    )
    pipe._opportunity_map = {}
    pipe._opportunity_gate = 0.0
    pipe._last_signal = None
    pipe._initial_weights = {}
    pipe._adds_per_position = {}
    pipe._rotations_this_month = 0
    pipe._rotations_month_key = ""
    pipe._v33_state_path = (
        state_path if state_path is not None
        else Path(tempfile.gettempdir()) / "v33_session_state_test.json"
    )
    pipe.executor = executor
    pipe.book_optimizer = SimpleNamespace(exit_thesis=exit_engine)
    return pipe


def _ctx(candidates):
    return DecisionContext(actions=[], signal=None, candidates=candidates)


def _candidate(ticker, net_edge=0.005):
    return EdgeCandidate(
        date=pd.Timestamp("2026-05-10"),
        ticker=ticker,
        raw_opportunity=0.003,
        direction=0.05,
        conviction=0.7,
        regime="neutral",
        regime_score=0.0,
        sector="tech",
        liquidity_bucket="high",
        vol_state="neutral",
        expected_return_5d=net_edge,
        net_edge_5d=net_edge,
        edge_tier="A",
    )


def _trigger_decision(should_exit, reason="profit_take"):
    return SimpleNamespace(should_exit=should_exit, reason=reason)


# ──────────────────────────────────────────────────────────────
# monitor() routing
# ──────────────────────────────────────────────────────────────
class TestMonitorRouting:
    def test_features_off_uses_v321_path(self, cfg_features_off):
        executor = MagicMock()
        executor.monitor_positions.return_value = []
        pipe = _make_pipeline(cfg_features_off, ctx=None, executor=executor,
                              exit_engine=None)
        pipe.monitor()
        executor.monitor_positions.assert_called_once()

    def test_features_on_but_ctx_none_uses_v321_path(self, cfg_features_on):
        executor = MagicMock()
        executor.monitor_positions.return_value = []
        engine = MagicMock(spec=ExitThesisEngine)
        pipe = _make_pipeline(cfg_features_on, ctx=None, executor=executor,
                              exit_engine=engine)
        pipe.monitor()
        executor.monitor_positions.assert_called_once()
        engine.decide.assert_not_called()

    def test_features_on_with_ctx_uses_v33_path(self, cfg_features_on):
        executor = MagicMock()
        executor.positions.positions = []  # no positions → early return
        engine = MagicMock(spec=ExitThesisEngine)
        ctx = _ctx([_candidate("AAPL")])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        pipe.monitor()
        executor.monitor_positions.assert_not_called()


# ──────────────────────────────────────────────────────────────
# _monitor_v33 behavior
# ──────────────────────────────────────────────────────────────
class TestMonitorV33:
    def _executor_with_positions(self, positions, price_info):
        executor = MagicMock()
        executor.positions.positions = positions
        executor._get_price.return_value = price_info
        return executor

    def test_no_positions_returns_empty(self, cfg_features_on):
        executor = self._executor_with_positions([], None)
        engine = MagicMock(spec=ExitThesisEngine)
        ctx = _ctx([_candidate("AAPL")])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        result = pipe._monitor_v33("2026-05-10")
        assert result == []
        engine.decide.assert_not_called()

    def test_no_trigger_skips_position(self, cfg_features_on):
        positions = [{
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 1, "entry_date": "2026-05-09",
            "entry_vol": 0.2, "confidence": 0.5,
        }]
        executor = self._executor_with_positions(
            positions, {"price": 205.0, "vol": 0.2, "low": 200.0},
        )
        executor.exit_rules.check.return_value = _trigger_decision(False)
        engine = MagicMock(spec=ExitThesisEngine)
        ctx = _ctx([_candidate("AAPL")])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        result = pipe._monitor_v33("2026-05-10")
        assert result == []
        engine.decide.assert_not_called()
        executor.execute_actions.assert_not_called()

    def test_trigger_fires_calls_exit_thesis_and_executes(self, cfg_features_on):
        positions = [{
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 3, "entry_date": "2026-05-07",
            "entry_vol": 0.2, "confidence": 0.5, "opportunity": 0.005,
        }]
        executor = self._executor_with_positions(
            positions, {"price": 215.0, "vol": 0.2, "low": 210.0},
        )
        executor.exit_rules.check.return_value = _trigger_decision(
            True, reason="profit_take",
        )
        executor.execute_actions.return_value = {
            "entered": [], "exited": [{
                "ticker": "AAPL", "reason": "vetoed:profit_take",
                "return": 0.075, "source": "exit_thesis",
            }], "trimmed": [], "rotated": [], "added": [],
        }

        engine = MagicMock(spec=ExitThesisEngine)
        action = BookAction(
            date=pd.Timestamp("2026-05-10"),
            action_type="EXIT", ticker="AAPL",
            target_weight=0.0, current_weight=0.3,
            reason="residual_edge_below_hold_min(0.0)",
            source_policy="exit_thesis",
        )
        engine.decide.return_value = action

        ctx = _ctx([_candidate("AAPL", net_edge=0.001)])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        result = pipe._monitor_v33("2026-05-10")

        engine.decide.assert_called_once()
        kwargs = engine.decide.call_args.kwargs
        assert kwargs["exit_trigger"] == "profit_take"
        assert kwargs["position"].ticker == "AAPL"
        assert kwargs["current_edge"].ticker == "AAPL"

        executor.execute_actions.assert_called_once()
        assert result[0]["ticker"] == "AAPL"

    def test_rotation_updates_cross_session_state(self, cfg_features_on):
        positions = [{
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 3, "entry_date": "2026-05-07",
            "entry_vol": 0.2, "confidence": 0.5,
        }]
        executor = self._executor_with_positions(
            positions, {"price": 210.0, "vol": 0.2, "low": 205.0},
        )
        executor.exit_rules.check.return_value = _trigger_decision(
            True, reason="profit_take",
        )
        executor.execute_actions.return_value = {
            "entered": [], "exited": [], "trimmed": [], "added": [],
            "rotated": [{
                "ticker_out": "AAPL", "ticker_in": "MSFT",
                "weight": 0.3, "reason": "rotation_edge",
            }],
        }
        engine = MagicMock(spec=ExitThesisEngine)
        engine.decide.return_value = BookAction(
            date=pd.Timestamp("2026-05-10"), action_type="ROTATE",
            ticker="AAPL", target_weight=0.0, current_weight=0.3,
            reason="rotation_edge", source_policy="exit_thesis",
            replacement_ticker="MSFT",
        )

        ctx = _ctx([_candidate("AAPL", 0.001), _candidate("MSFT", 0.008)])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        # Seed pre-existing tracking state
        pipe._initial_weights = {"AAPL": 0.3}
        pipe._adds_per_position = {"AAPL": 1}

        pipe._monitor_v33("2026-05-10")

        assert pipe._rotations_this_month == 1
        assert "AAPL" not in pipe._initial_weights
        assert pipe._initial_weights["MSFT"] == 0.3
        assert "AAPL" not in pipe._adds_per_position

    def test_replacement_pool_excludes_held_tickers(self, cfg_features_on):
        positions = [{
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 3, "entry_date": "2026-05-07",
            "entry_vol": 0.2, "confidence": 0.5,
        }]
        executor = self._executor_with_positions(
            positions, {"price": 210.0, "vol": 0.2, "low": 205.0},
        )
        executor.exit_rules.check.return_value = _trigger_decision(True)
        executor.execute_actions.return_value = {
            "entered": [], "exited": [], "trimmed": [],
            "rotated": [], "added": [],
        }
        engine = MagicMock(spec=ExitThesisEngine)
        engine.decide.return_value = BookAction(
            date=pd.Timestamp("2026-05-10"), action_type="KEEP",
            ticker="AAPL", target_weight=0.3, current_weight=0.3,
            reason="thesis_alive", source_policy="exit_thesis",
        )

        ctx = _ctx([
            _candidate("AAPL", 0.005),  # held — must NOT be in replacement pool
            _candidate("MSFT", 0.008),  # available
            _candidate("NVDA", 0.006),  # available
        ])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        pipe._monitor_v33("2026-05-10")

        replacement_pool = engine.decide.call_args.kwargs["replacement_candidates"]
        tickers = {c.ticker for c in replacement_pool}
        assert "AAPL" not in tickers
        assert tickers == {"MSFT", "NVDA"}


# ──────────────────────────────────────────────────────────────
# _convert_one_position_state
# ──────────────────────────────────────────────────────────────
class TestConvertOnePositionState:
    def test_pnl_positive(self):
        pipe = LivePipeline.__new__(LivePipeline)
        pos = {
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 2, "entry_date": "2026-05-08",
            "opportunity": 0.004,
        }
        state = pipe._convert_one_position_state(
            pos, current_price=210.0, today="2026-05-11",
        )
        assert state.ticker == "AAPL"
        assert state.entry_price == 200.0
        assert state.current_price == 210.0
        assert state.pnl_pct == pytest.approx(0.05)
        assert state.thesis_alive is True
        # calendar-day diff from entry_date 2026-05-08 → 2026-05-11 = 3
        # (NOT stored hold_days 2 + 1; both happen to be 3 here)
        assert state.holding_days == 3
        assert state.entry_edge == pytest.approx(0.004)

    def test_pnl_negative(self):
        pipe = LivePipeline.__new__(LivePipeline)
        pos = {"ticker": "X", "entry_price": 100.0, "weight": 0.2,
               "hold_days": 0, "entry_date": "2026-05-10"}
        state = pipe._convert_one_position_state(pos, current_price=95.0)
        assert state.pnl_pct == pytest.approx(-0.05)
        assert state.thesis_alive is False
        assert state.drawdown_from_peak_pct == pytest.approx(-0.05)

    def test_residual_edge_from_candidate(self):
        """Issue 4 fix — current_edge.net_edge_5d → PositionState.residual_edge."""
        pipe = LivePipeline.__new__(LivePipeline)
        pos = {"ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
               "hold_days": 1, "entry_date": "2026-05-09"}
        cand = _candidate("AAPL", net_edge=0.0042)
        state = pipe._convert_one_position_state(
            pos, current_price=205.0, current_edge=cand,
        )
        assert state.residual_edge == pytest.approx(0.0042)

    def test_residual_edge_default_zero_when_no_candidate(self):
        pipe = LivePipeline.__new__(LivePipeline)
        pos = {"ticker": "X", "entry_price": 100.0, "weight": 0.2,
               "hold_days": 0, "entry_date": "2026-05-10"}
        state = pipe._convert_one_position_state(pos, current_price=100.0)
        assert state.residual_edge == 0.0


# ──────────────────────────────────────────────────────────────
# Critical 1 fix — signal_refresh_fn callback
# ──────────────────────────────────────────────────────────────
class TestRefreshEdgeCandidate:
    def test_returns_candidate_when_in_ctx(self):
        pipe = LivePipeline.__new__(LivePipeline)
        cand = _candidate("AAPL", net_edge=0.005)
        pipe._last_ctx = _ctx([cand, _candidate("MSFT")])
        assert pipe._refresh_edge_candidate("AAPL") is cand

    def test_returns_none_when_no_ctx(self):
        pipe = LivePipeline.__new__(LivePipeline)
        pipe._last_ctx = None
        assert pipe._refresh_edge_candidate("AAPL") is None

    def test_returns_none_when_ticker_missing(self):
        pipe = LivePipeline.__new__(LivePipeline)
        pipe._last_ctx = _ctx([_candidate("MSFT")])
        assert pipe._refresh_edge_candidate("AAPL") is None


# ──────────────────────────────────────────────────────────────
# Critical 2 fix — V3.3 state persistence
# ──────────────────────────────────────────────────────────────
class TestV33StatePersistence:
    def _pipe_with_state_path(self, tmp_path):
        pipe = LivePipeline.__new__(LivePipeline)
        pipe._v33_state_path = tmp_path / "v33_session_state.json"
        pipe._adds_per_position = {}
        pipe._initial_weights = {}
        pipe._rotations_this_month = 0
        pipe._rotations_month_key = ""
        return pipe

    def test_save_then_load_round_trip(self, tmp_path):
        p1 = self._pipe_with_state_path(tmp_path)
        p1._adds_per_position = {"AAPL": 2, "MSFT": 1}
        p1._initial_weights = {"AAPL": 0.3, "MSFT": 0.25}
        p1._rotations_this_month = 3
        p1._rotations_month_key = "2026-05"
        p1._save_v33_state()

        p2 = self._pipe_with_state_path(tmp_path)
        p2._load_v33_state()
        assert p2._adds_per_position == {"AAPL": 2, "MSFT": 1}
        assert p2._initial_weights == {"AAPL": 0.3, "MSFT": 0.25}
        assert p2._rotations_this_month == 3
        assert p2._rotations_month_key == "2026-05"

    def test_load_missing_file_keeps_defaults(self, tmp_path):
        p = self._pipe_with_state_path(tmp_path)
        p._load_v33_state()
        assert p._adds_per_position == {}
        assert p._rotations_this_month == 0

    def test_load_malformed_recovers_silently(self, tmp_path):
        p = self._pipe_with_state_path(tmp_path)
        p._v33_state_path.write_text("{invalid json")
        p._load_v33_state()
        assert p._adds_per_position == {}

    def test_monitor_v33_persists_after_rotation(
        self, cfg_features_on, tmp_path,
    ):
        positions = [{
            "ticker": "AAPL", "entry_price": 200.0, "weight": 0.3,
            "hold_days": 3, "entry_date": "2026-05-07",
            "entry_vol": 0.2, "confidence": 0.5,
        }]
        executor = MagicMock()
        executor.positions.positions = positions
        executor._get_price.return_value = {
            "price": 210.0, "vol": 0.2, "low": 205.0,
        }
        executor.exit_rules.check.return_value = _trigger_decision(True)
        executor.execute_actions.return_value = {
            "entered": [], "exited": [], "trimmed": [], "added": [],
            "rotated": [{
                "ticker_out": "AAPL", "ticker_in": "MSFT",
                "weight": 0.3, "reason": "rotation_edge",
            }],
        }
        engine = MagicMock(spec=ExitThesisEngine)
        engine.decide.return_value = BookAction(
            date=pd.Timestamp("2026-05-10"), action_type="ROTATE",
            ticker="AAPL", target_weight=0.0, current_weight=0.3,
            reason="rotation_edge", source_policy="exit_thesis",
            replacement_ticker="MSFT",
        )
        ctx = _ctx([_candidate("AAPL", 0.001), _candidate("MSFT", 0.008)])
        pipe = _make_pipeline(cfg_features_on, ctx=ctx, executor=executor,
                              exit_engine=engine)
        # Inject persistence path
        pipe._v33_state_path = tmp_path / "v33_session_state.json"

        pipe._monitor_v33("2026-05-10")

        assert pipe._v33_state_path.exists()
        import json
        data = json.loads(pipe._v33_state_path.read_text())
        assert data["rotations_this_month"] == 1
        assert data["initial_weights"] == {"MSFT": 0.3}
