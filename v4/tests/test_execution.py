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


class TestOrderRetry:
    """일시적 KIS 에러(500/Connection aborted) 시 _order 재시도.

    2026-06-15 V4 첫 실거래 회귀: 20 picks 중 3종목(031330/089970/010170)이
    HTTP 500 / RemoteDisconnected(전부 일시적 통신 에러)로 거부됐는데 _order 가
    재시도 없이 즉시 None 반환 → 17/20만 체결. 6/17 프로브로 셋 다 정상 매매가능
    확인됨(=일시적, universe 문제 아님). 비일시적 에러(계좌차단 40910000)는 즉시
    실패 유지(6/1 동작 보존).
    """

    class FlakyKISApi:
        """fail_times 번 transient 예외 raise 후 성공. raise_msg 로 에러 종류 지정."""
        def __init__(self, fail_times: int, raise_msg: str = "500 Server Error: Internal Server Error"):
            self.fail_times = fail_times
            self.raise_msg = raise_msg
            self.calls = 0
            self.placed: list[tuple] = []

        def order_domestic(self, t, side, qty, order_type="01"):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise Exception(self.raise_msg)
            self.placed.append((side, t, qty))
            return {"order_no": "OK", "ticker": t, "side": side, "qty": qty}

    def _broker(self, tmp_path, api):
        b = KisKoreaBroker.__new__(KisKoreaBroker)
        b.api = api
        b.dry_run = False
        b.log_path = Path(tmp_path) / "trades.jsonl"
        b.RETRY_WAIT_S = 0.0          # 테스트 가속 (sleep 제거)
        return b

    def test_transient_500_retries_then_succeeds(self, tmp_path):
        api = self.FlakyKISApi(fail_times=2)               # 2번 실패 후 성공
        b = self._broker(tmp_path, api)
        r = b.buy_qty("031330", 100, ref_price=15_000)
        assert r is not None and r["order_no"] == "OK"
        assert api.calls == 3                              # 2 재시도 + 1 성공
        assert len(api.placed) == 1                        # 단 1회 체결 (중복 없음)

    def test_connection_aborted_retries(self, tmp_path):
        api = self.FlakyKISApi(fail_times=1, raise_msg="('Connection aborted.', RemoteDisconnected())")
        b = self._broker(tmp_path, api)
        assert b.buy_qty("089970", 50, ref_price=90_000) is not None
        assert api.calls == 2

    def test_persistent_transient_exhausts_and_fails(self, tmp_path):
        api = self.FlakyKISApi(fail_times=99)              # 계속 실패
        b = self._broker(tmp_path, api)
        assert b.buy_qty("010170", 200, ref_price=18_000) is None
        assert api.calls == KisKoreaBroker.MAX_ORDER_RETRIES   # 정해진 횟수만큼 시도 후 포기

    def test_non_transient_fails_fast_no_retry(self, tmp_path):
        # 계좌차단(40910000) = 비일시적 → 재시도 금지, 즉시 None (6/1 동작 보존)
        api = self.FlakyKISApi(fail_times=99,
                               raise_msg="모의투자 주문이 불가한 계좌입니다. (code=40910000)")
        b = self._broker(tmp_path, api)
        assert b.buy_qty("000087", 93, ref_price=13_000) is None
        assert api.calls == 1                              # 재시도 없음


class FlakyBroker(FakeBroker):
    """지정 ticker 의 buy/sell 이 None 반환 (KIS 매매불가/자금부족/장외 모사)."""

    def __init__(self, cash, positions, prices, fail=()):
        super().__init__(cash, positions, prices)
        self.fail = {self.norm(t) for t in fail}

    def buy_qty(self, ticker, qty, ref_price=0.0):
        if self.norm(ticker) in self.fail:
            return None
        return super().buy_qty(ticker, qty, ref_price)

    def sell(self, ticker, qty, ref_price=None):
        if self.norm(ticker) in self.fail:
            return None
        return super().sell(ticker, qty, ref_price)


class TestReconcileFailures:
    """주문 실패(broker None 반환)가 silent skip 되지 않고 plan.failures 로 surface 되는가.

    2026-06-07 V4 paper 6/1 silent execution failure 회귀: 우선주 매매불가 + 자금부족으로
    전 주문이 None 반환됐는데 executor 가 조용히 continue → 계좌 무변동인데 성공처럼 보임.
    """
    PRICES = {"100000": 10_000.0, "200000": 20_000.0, "400000": 8_000.0}

    def test_failed_exit_sell_recorded_in_failures(self):
        # 보유 400000 청산 주문이 실패 → sells 아님, failures 에 기록
        b = FlakyBroker(cash=0, positions={"400000": 100}, prices=self.PRICES,
                        fail=["400000"])
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert not any(o["ticker"] == "400000" for o in plan.sells)
        assert any(o["ticker"] == "400000" and o["side"] == "sell" for o in plan.failures)

    def test_failed_buy_recorded_in_failures(self):
        b = FlakyBroker(cash=10_000_000, positions={}, prices=self.PRICES,
                        fail=["100000"])
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert not plan.buys
        assert any(o["ticker"] == "100000" and o["side"] == "buy" for o in plan.failures)

    def test_all_orders_fail_executed_zero(self):
        # 6/1 시나리오: 청산도 매수도 전부 실패 → 성공 0, 실패 전부
        b = FlakyBroker(cash=10_000_000, positions={"400000": 100}, prices=self.PRICES,
                        fail=["400000", "100000"])
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert len(plan.sells) + len(plan.buys) == 0
        assert len(plan.failures) >= 2

    def test_partial_failure(self):
        # 청산 성공 + 매수 실패 → 부분
        b = FlakyBroker(cash=0, positions={"400000": 100}, prices=self.PRICES,
                        fail=["100000"])
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert any(o["ticker"] == "400000" for o in plan.sells)       # 청산 성공
        assert any(o["ticker"] == "100000" for o in plan.failures)    # 매수 실패

    def test_success_has_no_failures(self):
        b = FakeBroker(cash=10_000_000, positions={}, prices=self.PRICES)
        plan = rebalance(b, {"100000": 1.0}, self.PRICES)
        assert plan.failures == ()

    def test_plan_only_no_failures(self):
        # execute=False → broker 미호출 → 실패 없음 (계획만)
        b = FlakyBroker(cash=10_000_000, positions={}, prices=self.PRICES,
                        fail=["100000"])
        plan = rebalance(b, {"100000": 1.0}, self.PRICES, execute=False)
        assert plan.failures == ()
        assert len(plan.buys) == 1


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
