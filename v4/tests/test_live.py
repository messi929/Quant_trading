"""V4 Stage3 live runner 테스트 — rebalance-or-hold + vol-target history (오프라인).

synthetic 패널 + FakeBroker + 임시 state. 실 FDR/KIS 없이 runner 의사결정 로직 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v4.config import KoreaConfig
from v4.execution.kis_broker import KisKoreaBroker
from v4.live.state import LiveState
from v4.live.runner import run_session, is_rebalance_day, trading_days_since

CFG = KoreaConfig(min_candidates=5, liq_top=10, n_pos=3,
                  lookbacks=(5, 10), sma_window=20, vol_win=3, hold=5)


TICKERS = [f"{100000 + i:06d}" for i in range(30)]   # 실제 KRX처럼 6자리 숫자


def _panel(n_days: int = 60):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close_d, vol_d = {}, {}
    for i, c in enumerate(TICKERS):
        g = (0.012 - 0.003 * i) if i < 10 else 0.001
        close_d[c] = 100.0 * (1 + g) ** np.arange(n_days)
        vol_d[c] = np.full(n_days, 1e9 * (30 - i))
    close = pd.DataFrame(close_d, index=dates)
    dvol = close * pd.DataFrame(vol_d, index=dates)
    up = pd.Series(100.0 * 1.01 ** np.arange(n_days), index=dates)     # regime ON
    down = pd.Series(100.0 * 0.99 ** np.arange(n_days), index=dates)   # regime OFF
    return close, dvol, up, down


class FakeBroker:
    norm = staticmethod(KisKoreaBroker.norm)

    def __init__(self, cash, positions, prices):
        self.cash = cash
        self.positions = {self.norm(t): q for t, q in positions.items()}
        self.prices = {self.norm(t): p for t, p in prices.items()}
        self.orders = []

    def get_balance(self):
        pos = [{"ticker": t, "qty": q} for t, q in self.positions.items()]
        ev = self.cash + sum(q * self.prices[t] for t, q in self.positions.items())
        return {"cash": self.cash, "total_eval": ev, "positions": pos}

    def get_price_krw(self, t):
        return self.prices.get(self.norm(t), 0.0)

    def buy_qty(self, t, qty, ref_price=0.0):
        tn = self.norm(t); self.orders.append(("buy", tn, int(qty)))
        return {"order_no": "F", "ticker": tn, "qty": int(qty)}

    def sell(self, t, qty, ref_price=None):
        tn = self.norm(t); self.orders.append(("sell", tn, int(qty)))
        return {"order_no": "F", "ticker": tn, "qty": int(qty)}


def _prices(close):
    return {t: float(close.iloc[-1][t]) for t in close.columns}


class TestRebalanceDay:
    def test_first_run_is_rebalance(self):
        close, _, up, _ = _panel()
        assert is_rebalance_day(LiveState(), up, close.index[-1], CFG) is True

    def test_same_day_holds(self):
        close, _, up, _ = _panel()
        st = LiveState(last_rebalance_date=str(close.index[-1].date()))
        assert is_rebalance_day(st, up, close.index[-1], CFG) is False

    def test_after_hold_days_rebalances(self):
        close, _, up, _ = _panel()
        st = LiveState(last_rebalance_date=str(close.index[-(CFG.hold + 1)].date()))
        assert trading_days_since(up, st.last_rebalance_date, close.index[-1]) == CFG.hold
        assert is_rebalance_day(st, up, close.index[-1], CFG) is True


class TestRunSession:
    def test_first_rebalance_places_buys_and_sets_state(self):
        close, dvol, up, _ = _panel()
        b = FakeBroker(10_000_000, {}, _prices(close))
        st = LiveState()
        res = run_session(b, close, dvol, up, st, CFG)
        assert res.rebalanced and res.regime_on
        assert any(o[0] == "buy" for o in b.orders)
        assert st.last_rebalance_date == str(close.index[-1].date())
        assert st.pending_entries                       # phantom 픽 기록됨

    def test_hold_day_no_orders(self):
        close, dvol, up, _ = _panel()
        b = FakeBroker(10_000_000, {}, _prices(close))
        st = LiveState(last_rebalance_date=str(close.index[-1].date()))
        res = run_session(b, close, dvol, up, st, CFG)
        assert res.rebalanced is False and b.orders == []

    def test_gate_off_liquidates_to_cash(self):
        close, dvol, _, down = _panel()
        b = FakeBroker(0, {TICKERS[0]: 100}, _prices(close))
        res = run_session(b, close, dvol, down, LiveState(), CFG)
        assert res.regime_on is False and res.exposure == 0.0
        assert ("sell", TICKERS[0], 100) in b.orders          # 전량 청산

    def test_pending_measured_into_history(self):
        close, dvol, up, _ = _panel()
        b = FakeBroker(10_000_000, {}, _prices(close))
        entry = float(close.iloc[-1][TICKERS[0]]) / 1.10   # 직전 entry → 약 +10%
        st = LiveState(last_rebalance_date=str(close.index[-(CFG.hold + 1)].date()),
                       pending_entries={TICKERS[0]: entry})
        res = run_session(b, close, dvol, up, st, CFG)
        assert res.measured_basket_ret is not None
        assert len(st.basket_history) == 1
        assert abs(st.basket_history[0] - (0.10 - CFG.cost)) < 1e-6

    def test_high_vol_history_reduces_exposure(self):
        close, dvol, up, _ = _panel()
        b = FakeBroker(10_000_000, {}, _prices(close))
        st = LiveState(basket_history=[0.3, -0.3, 0.3, -0.3, 0.3, -0.3])  # 고변동
        res = run_session(b, close, dvol, up, st, CFG)
        assert 0.0 < res.exposure < 1.0                  # vol-target 축소
