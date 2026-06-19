"""V4 트렌치 live runner 회귀 (증분 2) — 합성 패널, 결정적 오프라인.

검증: 시차 부트스트랩(트렌치 t 는 t·step 거래일 후 첫 진입) / 결합 book = Σ 트렌치 /
20일 후 full 배치 / 전 주문 실패 시 미전진 / state 영속 round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.config import KoreaConfig
from v4.engine import ensemble_picks, regime_on
from v4.live.state import TrancheLiveState, TrancheSlot
from v4.live.tranche_runner import run_tranche_session
from v4.tests.test_v4 import _synthetic_panel
from v4.tests.test_execution import FakeBroker, FlakyBroker

CACHE = Path("v3/research/reports/korea_kosdaq_long_cache.parquet")
KQ11_FIXTURE = Path("v4/tests/fixtures/kq11_index.parquet")

# 합성 cfg: SMALL 과 동일하되 N=5 step=4 (hold20//5)
CFG = KoreaConfig(min_candidates=3, liq_top=10, n_pos=3, lookbacks=(5, 10),
                  sma_window=20, vol_win=3)
N = 5
STEP = CFG.hold // N           # 4


def _panel(n_days=140):
    close, dvol = _synthetic_panel(n_days)
    up = pd.Series(100.0 * 1.01 ** np.arange(len(close)), index=close.index)  # gate on
    return close, dvol, up


def _session(state, close, dvol, index, upto, *, execute=False, broker=None):
    c, d = close.iloc[:upto + 1], dvol.iloc[:upto + 1]
    i = len(c) - 1
    prices = {t: float(close.iloc[i][t]) for t in close.columns}
    b = broker if broker is not None else FakeBroker(1e8, {}, prices)
    return run_tranche_session(b, c, d, index, state, CFG, execute=execute), b


class TestBootstrapStagger:
    def test_first_session_only_tranche0(self):
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        res, _ = _session(st, close, dvol, idx, 60)
        assert res.rebalanced_tranches == (0,)
        assert st.anchor_date == str(close.index[60].date())
        assert st.tranches[0].last_rebalance_date is not None
        assert all(st.tranches[t].last_rebalance_date is None for t in range(1, N))

    def test_staggered_activation_daily(self):
        # anchor 부터 매 거래일 세션 → 트렌치 t 첫 진입이 elapsed==t·step 에 발생
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        start = 60
        first_rebal_elapsed = {}
        for k in range(0, 4 * STEP + 2):              # 0..17 거래일
            res, _ = _session(st, close, dvol, idx, start + k)
            for t in res.rebalanced_tranches:
                if t not in first_rebal_elapsed:
                    first_rebal_elapsed[t] = k
        for t in range(N):
            assert first_rebal_elapsed[t] == t * STEP, f"트렌치 {t} 첫 진입 elapsed {first_rebal_elapsed.get(t)}"

    def test_full_deployment_after_ramp(self):
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        start = 60
        last = None
        for k in range(0, 4 * STEP + 1):              # 16 거래일이면 5 트렌치 전부 활성
            last, _ = _session(st, close, dvol, idx, start + k)
        assert all(s.last_rebalance_date is not None for s in st.tranches)
        gross = sum(sum(s.target_weights.values()) for s in st.tranches)
        assert gross == pytest.approx(1.0, abs=1e-9)  # gate on·exp 1.0 × 5×(1/5)


class TestCombinedBook:
    def test_combined_is_sum_of_active_tranches(self):
        # 2 트렌치 활성(동일 픽) → 결합 종목 weight = 2 × 단일 트렌치 weight
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        start = 60
        for k in range(0, STEP + 1):                  # t0, t1 활성
            res, _ = _session(st, close, dvol, idx, start + k)
        active = [s for s in st.tranches if s.target_weights]
        assert len(active) == 2
        per = active[0].target_weights
        tick = next(iter(per))
        combined = sum(s.target_weights.get(tick, 0.0) for s in st.tranches)
        assert combined == pytest.approx(2 * per[tick], abs=1e-12)

    def test_position_overlap_dedup(self):
        # 합성: 모든 트렌치 동일 3종목 → 결합 고유 종목 = 3 (중복 합산)
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        start = 60
        res = None
        for k in range(0, 4 * STEP + 1):
            res, _ = _session(st, close, dvol, idx, start + k)
        assert res.n_positions == CFG.n_pos        # 3


class TestExecFailureInvariant:
    def test_all_fail_no_advance(self):
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        i = 60
        prices = {t: float(close.iloc[i][t]) for t in close.columns}
        flaky = FlakyBroker(1e8, {}, prices, fail=list(close.columns))   # 전 종목 실패
        res = run_tranche_session(flaky, close.iloc[:i + 1], dvol.iloc[:i + 1], idx, st,
                                  CFG, execute=True)
        assert res.note == "exec_failed"
        assert res.rebalanced_tranches == ()
        assert st.tranches[0].last_rebalance_date is None   # 미전진

    def test_success_advances(self):
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        res, _ = _session(st, close, dvol, idx, 60, execute=True)
        assert res.note == "rebalanced"
        assert st.tranches[0].last_rebalance_date is not None


class TestStatePersistence:
    def test_roundtrip(self, tmp_path):
        close, dvol, idx = _panel()
        st = TrancheLiveState(n_tranches=N)
        for k in range(0, STEP + 1):
            _session(st, close, dvol, idx, 60 + k)
        p = tmp_path / "tranche_state.json"
        st.save(p)
        st2 = TrancheLiveState.load(p, n_tranches=N)
        assert st2.anchor_date == st.anchor_date
        assert st2.n_tranches == N
        assert [s.last_rebalance_date for s in st2.tranches] == \
               [s.last_rebalance_date for s in st.tranches]
        assert st2.tranches[0].target_weights == st.tranches[0].target_weights

    def test_missing_file_fresh(self, tmp_path):
        st = TrancheLiveState.load(tmp_path / "nope.json", n_tranches=N)
        assert st.anchor_date is None
        assert len(st.tranches) == N
        assert all(s.last_rebalance_date is None for s in st.tranches)

    def test_legacy_single_state_ignored(self, tmp_path):
        # 구버전 single LiveState 스키마 → fresh (clean cutover)
        p = tmp_path / "old.json"
        p.write_text('{"last_rebalance_date": "2026-06-15", "basket_history": [0.1]}',
                     encoding="utf-8")
        st = TrancheLiveState.load(p, n_tranches=N)
        assert st.anchor_date is None and len(st.tranches) == N


@pytest.mark.skipif(not (CACHE.exists() and KQ11_FIXTURE.exists()),
                    reason="survivorship-free 캐시(gitignore) 또는 KQ11 fixture 없음")
class TestLiveBacktestParity:
    """라이브 runner 를 실 캐시 위로 일별 replay → 각 트렌치 book 이 engine ensemble_picks
    와 일치(orchestration == 검증된 엔진). 곡선 단위가 아닌 book(종목) 단위 parity —
    정수/현금/dust 같은 체결 마찰을 배제하고 *로직* parity 를 증명. cfg 는 실 SPEC(N=5).
    """
    CFG = KoreaConfig()        # 실 SPEC: N=5, n_pos=20, lookbacks 40-120, sma200

    @staticmethod
    def _load():
        from v4.data import load_panel
        close, dvol = load_panel(CACHE)
        idx = pd.read_parquet(KQ11_FIXTURE)["Close"]
        idx.index = pd.to_datetime(idx.index).normalize()
        return close, dvol, idx

    def _replay(self, close, dvol, idx, n_sessions=45):
        """마지막 n_sessions 거래일을 일별 replay (부트스트랩 16일 + 재순환 포함)."""
        st = TrancheLiveState(n_tranches=self.CFG.n_tranches)
        start = len(close) - n_sessions
        for i in range(start, len(close)):
            c, d = close.iloc[:i + 1], dvol.iloc[:i + 1]
            prices = {t: float(close.iloc[i][t]) for t in close.columns
                      if pd.notna(close.iloc[i][t])}
            run_tranche_session(FakeBroker(1e8, {}, prices), c, d, idx, st, self.CFG,
                                execute=False)
        return st

    def test_all_tranches_active_after_replay(self):
        close, dvol, idx = self._load()
        st = self._replay(close, dvol, idx)
        assert all(s.last_rebalance_date is not None for s in st.tranches)

    def test_each_tranche_book_matches_engine_picks(self):
        # 핵심 parity: 각 트렌치 book 종목 == 그 트렌치 rebalance 시점 ensemble_picks
        close, dvol, idx = self._load()
        st = self._replay(close, dvol, idx)
        checked = 0
        for slot in st.tranches:
            i_t = close.index.get_loc(pd.Timestamp(slot.last_rebalance_date))
            picks = ensemble_picks(close.iloc[:i_t + 1], dvol.iloc[:i_t + 1], i_t, self.CFG)
            on = regime_on(idx, close.index[i_t], self.CFG)
            if on and picks.tickers:
                assert set(slot.target_weights) == set(picks.tickers), \
                    f"트렌치 book {set(slot.target_weights)} != picks {set(picks.tickers)}"
                checked += 1
            else:
                assert slot.target_weights == {}      # gate off → 빈 book
        assert checked >= 1, "gate-on 트렌치가 하나도 없음 — parity 검증 불가"

    def test_each_tranche_equal_weight_capped(self):
        close, dvol, idx = self._load()
        st = self._replay(close, dvol, idx)
        for slot in st.tranches:
            w = list(slot.target_weights.values())
            if not w:
                continue
            assert len(set(round(x, 12) for x in w)) == 1            # equal-weight
            assert sum(w) <= self.CFG.vol_cap / self.CFG.n_tranches + 1e-9  # ≤ cap/N

    def test_combined_gross_within_cap(self):
        close, dvol, idx = self._load()
        st = self._replay(close, dvol, idx)
        combined: dict[str, float] = {}
        for slot in st.tranches:
            for tk, wv in slot.target_weights.items():
                combined[tk] = combined.get(tk, 0.0) + wv
        gross = sum(combined.values())
        assert 0.0 < gross <= self.CFG.vol_cap + 1e-9     # 결합 노출 (executor 가 ≤1 clamp)
