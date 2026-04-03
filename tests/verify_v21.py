"""V2.1 server verification script — run on server to validate all 7 fixes."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta


def main():
    print("=" * 60)
    print("V2.1 서버 검증 시작")
    print("=" * 60)

    # ── 1. Import 검증 ──
    from live.executor import TradingExecutor
    from scheduler.runner import TradingRunner
    from strategy.stock_signal import ConvictionSignalGenerator
    from pipeline.signal_pipeline import SignalPipeline
    print("[1/7] PASS: 모든 모듈 import 성공")

    # ── Shared mocks ──
    class MockTL:
        def log_trade(self, *a, **kw): pass
        def _compute_cumulative_mdd(self): return -0.08

    class MockAPI:
        def get_domestic_balance(self):
            return {"total_eval": 100_000_000, "positions": []}
        def get_domestic_price(self, code):
            return {"price": 50000}
        def order_domestic(self, **kw):
            return {"order_no": "TEST001"}

    cfg = {
        "trading": {
            "stop_loss_pct": 0.02, "profit_take_pct": 0.025,
            "profit_take_full_pct": 0.05, "time_exit_hours": 2,
            "time_exit_min_profit": 0.01, "max_hold_days": 3,
            "transaction_cost_rate": 0.006, "retry_wait_secs": 300,
            "volume_participation_max": 0.05,
        },
        "broker": {"mode": "sandbox", "paper_trading": False},
        "risk": {
            "daily_loss_limit": 0.015,
            "circuit_breaker": {
                "caution": 0.05, "warning": 0.10,
                "high": 0.20, "crisis": 0.30,
            },
        },
        "paths": {"trade_log_db": "tracking/trades.db"},
    }

    # ── 2. 서킷 브레이커 검증 ──
    ex = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())

    MockTL._compute_cumulative_mdd = lambda self: -0.08
    assert ex._circuit_breaker_scale() == 0.75

    MockTL._compute_cumulative_mdd = lambda self: -0.15
    assert ex._circuit_breaker_scale() == 0.50

    MockTL._compute_cumulative_mdd = lambda self: -0.25
    assert ex._circuit_breaker_scale() == 0.25

    MockTL._compute_cumulative_mdd = lambda self: -0.35
    assert ex._circuit_breaker_scale() == 0.0

    MockTL._compute_cumulative_mdd = lambda self: -0.03
    assert ex._circuit_breaker_scale() == 1.0
    print("[2/7] PASS: 서킷 브레이커 — 3%→1.0, 8%→0.75, 15%→0.50, 25%→0.25, 35%→0.0")

    # ── 3. 부분 이익실현 검증 ──
    ex2 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex2.open_positions["005930.KS"] = {
        "qty": 600, "entry_price": 50000,
        "entry_time": datetime.now() - timedelta(minutes=30),
        "weight": 0.30, "market": "domestic",
        "score": 0.08, "partial_taken": False,
    }

    class PriceAPI25(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 51250}  # +2.5%

    ex2.api = PriceAPI25()
    exits = ex2.monitor_positions()
    assert len(exits) == 1
    pos = ex2.open_positions.get("005930.KS")
    assert pos is not None and pos["qty"] == 300 and pos["partial_taken"] is True

    # No double partial
    exits2 = ex2.monitor_positions()
    assert len(exits2) == 0

    # +5% → full close
    class PriceAPI50(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 52500}

    ex2.api = PriceAPI50()
    exits3 = ex2.monitor_positions()
    assert len(exits3) == 1
    assert "005930.KS" not in ex2.open_positions
    print("[3/7] PASS: 부분 이익실현 — +2.5%→50%(600→300), 중복X, +5%→전량청산")

    # ── 4. 매도 실패 → 잔고 확인 ──
    class FailAPI(MockAPI):
        def order_domestic(self, **kw):
            raise RuntimeError("sandbox error")
        def get_domestic_balance(self):
            return {"total_eval": 100_000_000, "positions": []}
        def get_domestic_price(self, code):
            return {"price": 6700}

    ex3 = TradingExecutor(cfg, api=FailAPI(), trade_logger=MockTL())
    ex3.open_positions["001527.KS"] = {
        "qty": 2863, "entry_price": 6660,
        "entry_time": datetime.now() - timedelta(hours=3),
        "weight": 0.20, "market": "domestic",
        "score": 0.06, "partial_taken": False,
    }
    result = ex3._close_position(
        "001527.KS", ex3.open_positions["001527.KS"], 6700, "time_exit",
    )
    assert result is None
    assert "001527.KS" not in ex3.open_positions
    print("[4/7] PASS: 매도 실패 → 잔고 없음 확인 → ghost position 제거")

    # Verify: if balance HAS the position, keep it
    class FailButHasAPI(FailAPI):
        def get_domestic_balance(self):
            return {"total_eval": 100_000_000, "positions": [{"ticker": "001527"}]}

    ex3b = TradingExecutor(cfg, api=FailButHasAPI(), trade_logger=MockTL())
    ex3b.open_positions["001527.KS"] = {
        "qty": 2863, "entry_price": 6660,
        "entry_time": datetime.now() - timedelta(hours=3),
        "weight": 0.20, "market": "domestic",
        "score": 0.06, "partial_taken": False,
    }
    result = ex3b._close_position(
        "001527.KS", ex3b.open_positions["001527.KS"], 6700, "time_exit",
    )
    assert result is None
    assert "001527.KS" in ex3b.open_positions  # kept because balance confirms it
    print("[4/7] PASS+: 매도 실패 → 잔고 있음 확인 → position 유지 (다음 사이클 재시도)")

    # ── 5. Reconciliation 검증 ──
    ex4 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex4.open_positions["043260.KQ"] = {
        "qty": 601, "entry_price": 46000,
        "entry_time": datetime.now(),
        "weight": 0.29, "market": "domestic",
        "score": 0.08, "partial_taken": False,
    }

    new_positions = [
        {"ticker": "043260.KQ", "weight": 0.29, "score": 0.08, "market": "domestic"},
        {"ticker": "005930.KS", "weight": 0.30, "score": 0.07, "market": "domestic"},
    ]
    results = ex4.execute_entry(new_positions, portfolio_value=100_000_000)
    assert "043260.KQ" in ex4.open_positions
    assert ex4.open_positions["043260.KQ"]["qty"] == 601  # unchanged
    assert "005930.KS" in ex4.open_positions
    assert len(results) == 1  # only 005930 bought
    print("[5/7] PASS: Reconciliation — 043260 유지(재매수X), 005930만 매수")

    # Old position rebalanced out
    ex5 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex5.open_positions["OLD.KS"] = {
        "qty": 100, "entry_price": 50000,
        "entry_time": datetime.now(),
        "weight": 0.20, "market": "domestic",
        "score": 0.05, "partial_taken": False,
    }
    results5 = ex5.execute_entry(
        [{"ticker": "NEW.KS", "weight": 0.30, "score": 0.07, "market": "domestic"}],
        portfolio_value=100_000_000,
    )
    assert "OLD.KS" not in ex5.open_positions
    assert "NEW.KS" in ex5.open_positions
    print("[5/7] PASS+: Reconciliation — OLD 청산(rebalance) + NEW 매수")

    # ── 6. reset_daily 검증 ──
    ex6 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex6.open_positions["TEST.KS"] = {"qty": 100}
    ex6._stoploss_tickers.add("TEST")
    ex6._daily_pnl = -0.005
    ex6._daily_traded_amount = 50_000_000
    ex6.reset_daily()
    assert "TEST.KS" in ex6.open_positions
    assert len(ex6._stoploss_tickers) == 0
    assert ex6._daily_pnl == 0.0
    assert ex6._daily_traded_amount == 0.0
    print("[6/7] PASS: reset_daily — positions 유지, counters 초기화")

    # ── 7. score_history 영속화 ──
    tmp = tempfile.mktemp(suffix=".json")
    gen = ConvictionSignalGenerator(score_history_path=tmp)
    assert gen._score_history == []

    scores = {
        "A": {"score": 0.05, "confidence": 0.6},
        "B": {"score": 0.03, "confidence": 0.5},
    }
    gen.generate(scores)
    assert os.path.exists(tmp)
    saved = json.loads(open(tmp).read())
    assert len(saved) == 2

    gen2 = ConvictionSignalGenerator(score_history_path=tmp)
    assert len(gen2._score_history) == 2
    os.remove(tmp)
    print("[7/7] PASS: score_history — 저장 + 로드 + 재시작 후 복구")

    print()
    print("=" * 60)
    print("V2.1 전체 검증 완료 — 7/7 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
