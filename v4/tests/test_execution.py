"""V4 Stage2 execution 테스트 — KIS broker norm + executor reconcile (오프라인 mock).

실제 KIS 연동은 KR 장중 + 네트워크 필요 → smoke 스크립트(v4/scripts/kis_smoke.py)로
별도. 여기선 reconcile 로직을 FakeBroker 로 결정적 검증.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from v4.execution.kis_broker import KisKoreaBroker
from v4.execution.executor import rebalance
from v4.scripts.reset_sandbox_account import reset


class FakeBroker:
    """executor 인터페이스(get_balance/get_price_krw/buy/sell/norm) mock."""
    norm = staticmethod(KisKoreaBroker.norm)

    def __init__(self, cash: float, positions: dict[str, int], prices: dict[str, float]):
        self.cash = cash
        self.positions = {self.norm(t): q for t, q in positions.items()}
        self.prices = {self.norm(t): p for t, p in prices.items()}
        self.orders: list[tuple[str, str, int]] = []

    def get_balance(self) -> dict:
        pos = [{"ticker": t, "qty": q} for t, q in self.positions.items()]
        eval_ = self.cash + sum(q * self.prices[t] for t, q in self.positions.items())
        return {"cash": self.cash, "total_eval": eval_, "positions": pos}

    def get_price_krw(self, ticker: str) -> float:
        return self.prices.get(self.norm(ticker), 0.0)

    def buy_qty(self, ticker: str, qty: int, ref_price: float = 0.0) -> dict:
        t = self.norm(ticker)
        self.orders.append(("buy", t, int(qty)))
        return {"order_no": "FAKE", "ticker": t, "qty": int(qty)}

    def sell(self, ticker: str, qty: int, ref_price=None) -> dict:
        t = self.norm(ticker)
        self.orders.append(("sell", t, int(qty)))
        return {"order_no": "FAKE", "ticker": t, "qty": int(qty)}


class TestNorm:
    def test_six_digit_passthrough(self):
        assert KisKoreaBroker.norm("005930") == "005930"

    def test_pads_leading_zeros(self):
        assert KisKoreaBroker.norm("5930") == "005930"

    def test_strips_suffix(self):
        assert KisKoreaBroker.norm("000660.KQ") == "000660"
        assert KisKoreaBroker.norm("035720.KS") == "035720"


class TestReconcile:
    PRICES = {"100000": 10_000.0, "200000": 20_000.0, "300000": 5_000.0, "400000": 8_000.0}

    def test_exit_unlisted_position(self):
        # 보유 400000 이 target 에 없음 → 전량 청산
        b = FakeBroker(cash=0, positions={"400000": 100}, prices=self.PRICES)
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert ("sell", "400000", 100) in b.orders
        assert any(o["ticker"] == "400000" and o["reason"] == "exit" for o in plan.sells)

    def test_new_buy_sizing(self):
        # 100% → 100000(가격 1만) on 10,000,000 → 1000주
        b = FakeBroker(cash=10_000_000, positions={}, prices=self.PRICES)
        rebalance(b, {"100000": 1.0}, self.PRICES)
        assert ("buy", "100000", 1000) in b.orders

    def test_equal_weight_two_names(self):
        b = FakeBroker(cash=10_000_000, positions={}, prices=self.PRICES)
        rebalance(b, {"100000": 0.5, "200000": 0.5}, self.PRICES)  # 각 5,000,000
        assert ("buy", "100000", 500) in b.orders   # 5M/10000
        assert ("buy", "200000", 250) in b.orders   # 5M/20000

    def test_no_margin_clamp(self):
        # 합 1.5 (cap leverage) → 1.0 으로 clamp
        b = FakeBroker(cash=10_000_000, positions={}, prices=self.PRICES)
        plan = rebalance(b, {"100000": 0.75, "200000": 0.75}, self.PRICES)
        assert plan.clamped
        spent = sum(q * self.PRICES[t] for _, t, q in b.orders if _ == "buy")
        assert spent <= plan.total_eval + 1e-6      # 무레버리지: 총매수 ≤ 평가액

    def test_dust_skip(self):
        # 이미 target 비중 거의 충족 → 미세 조정 skip
        b = FakeBroker(cash=0, positions={"100000": 1000}, prices=self.PRICES)
        plan = rebalance(b, {"100000": 1.0}, self.PRICES, min_order_krw=500_000)
        # 1000주 보유 = 10M, target 10M → diff 0 → 주문 없음
        assert not b.orders

    def test_trim_overweight(self):
        # 보유 1000주(10M)인데 target 50% (5M=500주) → 500주 매도
        b = FakeBroker(cash=0, positions={"100000": 1000}, prices=self.PRICES)
        rebalance(b, {"100000": 0.5}, self.PRICES)
        assert ("sell", "100000", 500) in b.orders

    def test_long_only_no_negative_qty(self):
        b = FakeBroker(cash=5_000_000, positions={"200000": 50}, prices=self.PRICES)
        rebalance(b, {"100000": 0.5, "300000": 0.5}, self.PRICES)
        assert all(q > 0 for _, _, q in b.orders)   # 음수 수량 없음

    def test_plan_only_no_execute(self):
        b = FakeBroker(cash=10_000_000, positions={}, prices=self.PRICES)
        plan = rebalance(b, {"100000": 1.0}, self.PRICES, execute=False)
        assert b.orders == []                        # 미주문
        assert len(plan.buys) == 1                    # 계획엔 존재


class TestBrokerSizing:
    """KisKoreaBroker buy/sell 수량 + dry_run (KISApi __init__ 우회, fake api 주입)."""

    class FakeKISApi:
        def __init__(self, prices): self.prices = prices; self.placed = []
        def get_domestic_price(self, t): return {"price": self.prices[t]}
        def order_domestic(self, t, side, qty, order_type="01"):
            self.placed.append((side, t, qty))
            return {"order_no": "X", "ticker": t, "side": side, "qty": qty}  # 실제 API 형식

    def _broker(self, tmp_path, dry_run, prices):
        b = KisKoreaBroker.__new__(KisKoreaBroker)
        b.api = self.FakeKISApi(prices)
        b.dry_run = dry_run
        b.log_path = Path(tmp_path) / "trades.jsonl"
        return b

    def test_buy_qty_from_amount(self, tmp_path):
        b = self._broker(tmp_path, dry_run=False, prices={"005930": 70_000})
        r = b.buy("005930", 700_000)
        assert r["qty"] == 10
        assert ("buy", "005930", 10) in b.api.placed

    def test_buy_below_one_share_returns_none(self, tmp_path):
        b = self._broker(tmp_path, dry_run=False, prices={"005930": 70_000})
        assert b.buy("005930", 50_000) is None        # < 1주

    def test_dry_run_places_no_order(self, tmp_path):
        b = self._broker(tmp_path, dry_run=True, prices={"005930": 70_000})
        r = b.buy("005930", 700_000)
        assert r["dry_run"] is True and r["qty"] == 10
        assert b.api.placed == []                      # 실주문 없음


class TestAccountReset:
    """reset_sandbox_account.reset — 잔여 전량 청산 로직 (sandbox 가동 전 #2)."""
    PRICES = {"000087": 13_250.0, "001420": 2_835.0, "001527": 7_400.0}

    def test_sells_all_positions(self):
        b = FakeBroker(cash=1_000_000,
                       positions={"000087": 93, "001420": 13532, "001527": 1655},
                       prices=self.PRICES)
        reset(b, execute=True)
        assert sorted(b.orders) == sorted([
            ("sell", "000087", 93), ("sell", "001420", 13532), ("sell", "001527", 1655)])

    def test_no_positions_is_noop(self):
        b = FakeBroker(cash=100_000_000, positions={}, prices={})
        assert reset(b, execute=True) == 0
        assert b.orders == []

    def test_skips_zero_qty(self):
        b = FakeBroker(cash=0, positions={"000087": 0, "001420": 100},
                       prices={"000087": 100.0, "001420": 100.0})
        reset(b, execute=True)
        assert b.orders == [("sell", "001420", 100)]
