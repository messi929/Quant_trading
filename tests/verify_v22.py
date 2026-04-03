"""V2.2 server verification — 3-day hold strategy."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta


def main():
    print("=" * 60)
    print("V2.2 서버 검증 (3일 보유 전략)")
    print("=" * 60)

    # ── 1. Import ──
    from live.executor import TradingExecutor
    from scheduler.runner import TradingRunner
    from strategy.stock_signal import ConvictionSignalGenerator
    print("[1/8] PASS: 모듈 import")

    # ── Mocks ──
    class MockTL:
        def log_trade(self, *a, **kw): pass
        def _compute_cumulative_mdd(self): return -0.03

    class MockAPI:
        def get_domestic_balance(self):
            return {"total_eval": 100_000_000, "positions": []}
        def get_domestic_price(self, code):
            return {"price": 50000}
        def order_domestic(self, **kw):
            return {"order_no": "T001"}

    cfg = {
        "trading": {
            "profit_take_full_pct": 0.05, "max_hold_days": 3,
            "transaction_cost_rate": 0.006, "volume_participation_max": 0.05,
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

    # ── 2. No stop_loss, no time_exit ──
    ex = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    assert not hasattr(ex, "stop_loss_pct")
    assert not hasattr(ex, "time_exit_hours")
    print("[2/8] PASS: stop_loss, time_exit 속성 없음")

    # ── 3. -3% drop → position kept (no individual stop) ──
    ex.open_positions["043260.KQ"] = {
        "qty": 601, "entry_price": 46000,
        "entry_time": datetime.now() - timedelta(hours=1),
        "weight": 0.29, "market": "domestic", "score": 0.08,
    }

    class PriceDrop(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 44620}  # -3%

    ex.api = PriceDrop()
    exits = ex.monitor_positions()
    assert len(exits) == 0
    assert "043260.KQ" in ex.open_positions
    print("[3/8] PASS: -3% drop → 개별 스탑 없음, 포지션 유지")

    # ── 4. 3h +0.5% → NOT time-exited ──
    ex2 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex2.open_positions["001527.KS"] = {
        "qty": 2863, "entry_price": 50000,
        "entry_time": datetime.now() - timedelta(hours=3),
        "weight": 0.20, "market": "domestic", "score": 0.06,
    }

    class PriceUp05(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 50250}  # +0.5%

    ex2.api = PriceUp05()
    exits2 = ex2.monitor_positions()
    assert len(exits2) == 0
    assert "001527.KS" in ex2.open_positions
    print("[4/8] PASS: 3시간 +0.5% → time_exit 없음, 보유 지속")

    # ── 5. +5% → profit_take_full ──
    class PriceUp5(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 52500}

    ex2.api = PriceUp5()
    exits3 = ex2.monitor_positions()
    assert len(exits3) == 1
    assert "001527.KS" not in ex2.open_positions
    print("[5/8] PASS: +5% → profit_take_full 청산")

    # ── 6. Session close: only expired, carry overnight ──
    ex3 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex3.open_positions["EXPIRED.KS"] = {
        "qty": 100, "entry_price": 50000,
        "entry_time": datetime.now() - timedelta(days=3),
        "weight": 0.30, "market": "domestic", "score": 0.07,
    }
    ex3.open_positions["KEEP.KS"] = {
        "qty": 200, "entry_price": 50000,
        "entry_time": datetime.now() - timedelta(days=1),
        "weight": 0.20, "market": "domestic", "score": 0.05,
    }
    exits4 = ex3.execute_exit()
    assert "EXPIRED.KS" not in ex3.open_positions
    assert "KEEP.KS" in ex3.open_positions
    assert len(exits4) == 1
    print("[6/8] PASS: session_close → 3일 만료만 청산, 1일 보유 유지")

    # ── 7. Portfolio loss limit (unrealized) ──
    ex4 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex4.open_positions["CRASH.KS"] = {
        "qty": 100, "entry_price": 50000,
        "entry_time": datetime.now(),
        "weight": 0.40, "market": "domestic", "score": 0.08,
    }

    class PriceCrash(MockAPI):
        def get_domestic_price(self, code):
            return {"price": 48000}  # -4% × 40% weight = -1.6%

    ex4.api = PriceCrash()
    exits5 = ex4.monitor_positions()
    assert "CRASH.KS" not in ex4.open_positions
    print("[7/8] PASS: 포트폴리오 미실현 -1.6% > 한도 -1.5% → force_close_all")

    # ── 8. Reconciliation + circuit breaker ──
    MockTL._compute_cumulative_mdd = lambda self: -0.08  # 8% MDD
    ex5 = TradingExecutor(cfg, api=MockAPI(), trade_logger=MockTL())
    ex5.open_positions["OLD.KS"] = {
        "qty": 100, "entry_price": 50000,
        "entry_time": datetime.now(),
        "weight": 0.20, "market": "domestic", "score": 0.05,
    }
    new_signal = [
        {"ticker": "OLD.KS", "weight": 0.20, "score": 0.05, "market": "domestic"},
        {"ticker": "NEW.KS", "weight": 0.30, "score": 0.07, "market": "domestic"},
    ]
    results = ex5.execute_entry(new_signal, portfolio_value=100_000_000)
    assert "OLD.KS" in ex5.open_positions  # kept
    assert "NEW.KS" in ex5.open_positions  # bought
    # Check circuit breaker scaled the weight: 0.30 * 0.75 = 0.225
    assert ex5.open_positions["NEW.KS"]["weight"] == 0.30 * 0.75
    print("[8/8] PASS: reconciliation(OLD 유지, NEW 매수) + 서킷브레이커 75% 스케일")

    # ── Config validation ──
    from config.config_loader import load_config
    cfg_full = load_config()
    assert "stop_loss_pct" not in cfg_full["trading"]
    assert "time_exit_hours" not in cfg_full["trading"]
    assert cfg_full["trading"]["profit_take_full_pct"] == 0.05
    assert cfg_full["trading"]["max_hold_days"] == 3
    assert cfg_full["schedule"]["kr"]["execute_entry"] == "09:30"
    assert cfg_full["project"]["version"] == "2.2.0"
    print()
    print("[CFG] PASS: config 정합성 확인 (v2.2.0, entry=09:30, no stop/time_exit)")

    print()
    print("=" * 60)
    print("V2.2 전체 검증 완료 — 8/8 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
