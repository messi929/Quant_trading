"""V4 트렌치 live runner 회귀 (증분 2) — 합성 패널, 결정적 오프라인.

검증: 시차 부트스트랩(트렌치 t 는 t·step 거래일 후 첫 진입) / 결합 book = Σ 트렌치 /
20일 후 full 배치 / 전 주문 실패 시 미전진 / state 영속 round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.config import KoreaConfig
from v4.live.state import TrancheLiveState, TrancheSlot
from v4.live.tranche_runner import run_tranche_session
from v4.tests.test_v4 import _synthetic_panel
from v4.tests.test_execution import FakeBroker, FlakyBroker

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
