"""V4 트렌치 백테스트 회귀 (증분 1).

invariant: offset 산출 + N=1 단일 phase 환원 + 합성 패널 결정적 불변식.
parity   : 실제 캐시 + KQ11 fixture → 트렌치 N=5 가 research(Sharpe~0.56/MDD−20%) 재현
           + 단일 phase 평균 대비 window luck 거세(MDD 개선) 확인. 캐시 없으면 skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.config import KoreaConfig
from v4.tranche import (
    tranche_offsets, phase_daily_returns, run_tranche_backtest, _daily_stats,
)
from v4.tests.test_v4 import _synthetic_panel, SMALL

CACHE = Path("v3/research/reports/korea_kosdaq_long_cache.parquet")
KQ11_FIXTURE = Path("v4/tests/fixtures/kq11_index.parquet")


class TestTrancheOffsets:
    def test_even_spacing_n5(self):
        assert tranche_offsets(5, 20) == [0, 4, 8, 12, 16]

    def test_n1_is_single_offset_zero(self):
        assert tranche_offsets(1, 20) == [0]

    def test_offsets_within_hold(self):
        for n in (2, 3, 4, 5, 6):
            assert all(0 <= o < 20 for o in tranche_offsets(n, 20))


class TestSyntheticInvariants:
    def test_n1_blend_equals_single_phase(self):
        # N=1 → 블렌드 = offset 0 단일 phase (정확히 동일)
        close, dvol = _synthetic_panel()
        up = pd.Series(100.0 * 1.01 ** np.arange(len(close)), index=close.index)
        res = run_tranche_backtest(close, dvol, up, SMALL, n_tranches=1)
        single = phase_daily_returns(close, dvol, up, 0, SMALL)
        assert res.offsets == (0,)
        assert res.sharpe == pytest.approx(_daily_stats(single)[0], abs=1e-12)

    def test_blend_series_finite_and_aligned(self):
        close, dvol = _synthetic_panel()
        up = pd.Series(100.0 * 1.01 ** np.arange(len(close)), index=close.index)
        for o in tranche_offsets(5, SMALL.hold):
            s = phase_daily_returns(close, dvol, up, o, SMALL)
            assert len(s) == len(close)
            assert np.all(np.isfinite(s.values))

    def test_uptrend_positive_return(self):
        # 전 종목 상승 + gate on → 트렌치 블렌드 양수
        close, dvol = _synthetic_panel()
        up = pd.Series(100.0 * 1.01 ** np.arange(len(close)), index=close.index)
        res = run_tranche_backtest(close, dvol, up, SMALL, n_tranches=5)
        assert res.total_return > 0

    def test_gate_off_all_cash(self):
        # 지수 하락 → 전 phase gate off → 수익 전부 0
        close, dvol = _synthetic_panel()
        down = pd.Series(100.0 * 0.99 ** np.arange(len(close)), index=close.index)
        res = run_tranche_backtest(close, dvol, down, SMALL, n_tranches=5)
        assert res.total_return == 0.0 and res.sharpe == 0.0


@pytest.mark.skipif(not (CACHE.exists() and KQ11_FIXTURE.exists()),
                    reason="survivorship-free 캐시(gitignore) 또는 KQ11 fixture 없음")
class TestFullCycleParity:
    """트렌치 N=5 = research korea_tranche.py 재현 + window luck 거세."""

    @staticmethod
    def _load():
        from v4.data import load_panel
        close, dvol = load_panel(CACHE)
        idx = pd.read_parquet(KQ11_FIXTURE)["Close"]
        idx.index = pd.to_datetime(idx.index).normalize()
        return close, dvol, idx

    def test_tranche5_reproduces_research(self):
        close, dvol, idx = self._load()
        r = run_tranche_backtest(close, dvol, idx, KoreaConfig(), n_tranches=5)
        assert 0.48 <= r.sharpe <= 0.64, f"tranche N=5 Sharpe {r.sharpe:.3f} != ~0.56"
        assert 0.15 <= abs(r.mdd) <= 0.25, f"tranche N=5 MDD {r.mdd:.3f} != ~-0.20"

    def test_window_luck_dispersion_present(self):
        # 단일 phase 분산이 실재(트렌치 정당화 근거): Sharpe 스프레드 유의
        close, dvol, idx = self._load()
        r = run_tranche_backtest(close, dvol, idx, KoreaConfig(), n_tranches=20)
        spread = max(r.phase_sharpes) - min(r.phase_sharpes)
        assert spread >= 0.20, f"phase Sharpe 스프레드 {spread:.2f} — window luck 미미?"

    def test_blend_mdd_better_than_worst_phase(self):
        # 블렌드 MDD 가 최악 단일 phase 보다 얕음 (리스크 평활)
        close, dvol, idx = self._load()
        r = run_tranche_backtest(close, dvol, idx, KoreaConfig(), n_tranches=5)
        worst_phase_mdd = min(r.phase_mdds)            # 가장 깊은(음수 큰) phase
        assert r.mdd > worst_phase_mdd, "블렌드 MDD 가 최악 phase 보다 깊으면 안 됨"

    def test_blend_sharpe_at_least_phase_mean(self):
        # 분산효과: 블렌드 Sharpe ≥ 개별 phase 평균 (− 작은 tol)
        close, dvol, idx = self._load()
        r = run_tranche_backtest(close, dvol, idx, KoreaConfig(), n_tranches=5)
        assert r.sharpe >= np.mean(r.phase_sharpes) - 0.05
