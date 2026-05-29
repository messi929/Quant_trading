"""V4 Korea engine 회귀 테스트.

invariant: 합성 패널 (오프라인·결정적) — 엔진 순수 함수 불변식.
parity   : 실제 survivorship-free 캐시 + KQ11 fixture → full-cycle Sharpe ~0.66 재현.
           캐시(gitignore) 없으면 skip (환경 의존, 로직 우회 아님).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.config import KoreaConfig
from v4.engine import (
    ensemble_picks, regime_on, vol_target_exposure, target_exposure, target_book,
)
from v4.backtest import run_backtest

CACHE = Path("v3/research/reports/korea_kosdaq_long_cache.parquet")
KQ11_FIXTURE = Path("v4/tests/fixtures/kq11_index.parquet")

# 합성 cfg: pool=top10, 후보≥5, N=3, 짧은 lookback (테스트용)
SMALL = KoreaConfig(min_candidates=5, liq_top=10, n_pos=3,
                    lookbacks=(5, 10), sma_window=20, vol_win=3)


def _synthetic_panel(n_days: int = 80):
    """30 ticker. T00~T09 = 고거래대금 pool, 그 중 T00~T04 양수추세 / T05~T09 음수.
    T10~T29 = 저거래대금(pool 밖). momentum 강도 T00 최고."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols, close_data, vol_data = [], {}, {}
    for i in range(30):
        g = (0.012 - 0.003 * i) if i < 10 else 0.001   # T00:+0.012 ... T05:-0.003 ...
        c = f"T{i:02d}"; cols.append(c)
        close_data[c] = 100.0 * (1 + g) ** np.arange(n_days)
        vol_data[c] = np.full(n_days, 1e9 * (30 - i))  # 거래량 i 증가→감소 (pool=T00~09)
    close = pd.DataFrame(close_data, index=dates)
    dvol = close * pd.DataFrame(vol_data, index=dates)
    return close, dvol


class TestEnsemblePicks:
    def test_respects_n_pos(self):
        close, dvol = _synthetic_panel()
        picks = ensemble_picks(close, dvol, 40, SMALL)
        assert len(picks.tickers) <= SMALL.n_pos

    def test_picks_within_pit_pool(self):
        close, dvol = _synthetic_panel()
        picks = ensemble_picks(close, dvol, 40, SMALL)
        pool = set(dvol.iloc[40].nlargest(SMALL.liq_top).index)
        assert set(picks.tickers) <= pool

    def test_trend_filter_excludes_negative_momentum(self):
        close, dvol = _synthetic_panel()
        picks = ensemble_picks(close, dvol, 40, SMALL)
        # T05~T09 음수추세 → 절대 선정 안 됨
        assert not (set(picks.tickers) & {f"T0{i}" for i in range(5, 10)})
        # 강한 추세 T00 가 포함 (top momentum)
        assert "T00" in picks.tickers

    def test_below_min_candidates_returns_empty(self):
        close, dvol = _synthetic_panel()
        cfg = KoreaConfig(min_candidates=100, liq_top=10, n_pos=3, lookbacks=(5, 10))
        assert ensemble_picks(close, dvol, 40, cfg).tickers == ()


class TestRegimeGate:
    def test_uptrend_on_downtrend_off(self):
        dates = pd.bdate_range("2020-01-01", periods=60)
        up = pd.Series(100.0 * (1.01) ** np.arange(60), index=dates)
        down = pd.Series(100.0 * (0.99) ** np.arange(60), index=dates)
        assert regime_on(up, dates[55], SMALL) is True
        assert regime_on(down, dates[55], SMALL) is False

    def test_returns_bool_and_handles_nan(self):
        dates = pd.bdate_range("2020-01-01", periods=60)
        idx = pd.Series(100.0 * (1.01) ** np.arange(60), index=dates)
        # 데이터 시작 이전 시각 → SMA NaN → off
        assert regime_on(idx, pd.Timestamp("2019-01-01"), SMALL) is False


class TestVolTarget:
    def test_short_history_returns_one(self):
        assert vol_target_exposure([0.01, 0.02], SMALL) == 1.0

    def test_zero_vol_returns_cap(self):
        assert vol_target_exposure([0.01] * 6, SMALL) == SMALL.vol_cap

    def test_within_bounds_and_monotone(self):
        cfg = KoreaConfig()
        lo = vol_target_exposure([0.01, -0.01, 0.01, -0.01, 0.01, -0.01], cfg)   # 저변동
        hi = vol_target_exposure([0.3, -0.3, 0.3, -0.3, 0.3, -0.3], cfg)         # 고변동
        assert 0.0 <= hi <= cfg.vol_cap and 0.0 <= lo <= cfg.vol_cap
        assert hi < lo          # 고변동일수록 노출 작음

    def test_regime_off_zero_exposure(self):
        # gate off → vol history 좋아도 노출 0
        assert target_exposure(False, [0.0] * 6, KoreaConfig()) == 0.0


class TestTargetBook:
    def test_empty_when_gate_off(self):
        close, dvol = _synthetic_panel()
        down_index = pd.Series(100.0 * (0.99) ** np.arange(len(close)), index=close.index)
        book = target_book(close, dvol, down_index, 40, [], SMALL)
        assert book == {}

    def test_weights_equal_and_sum_to_exposure(self):
        close, dvol = _synthetic_panel()
        up_index = pd.Series(100.0 * (1.01) ** np.arange(len(close)), index=close.index)
        book = target_book(close, dvol, up_index, 40, [], SMALL)  # 짧은 history → exp=1.0
        assert book, "gate on + 픽 존재 → 비어있으면 안 됨"
        assert abs(sum(book.values()) - 1.0) < 1e-9        # 총 노출 = exposure(1.0)
        assert len(set(round(w, 12) for w in book.values())) == 1   # equal weight
        assert all(t.startswith("T") for t in book)


@pytest.mark.skipif(not (CACHE.exists() and KQ11_FIXTURE.exists()),
                    reason="survivorship-free 캐시(gitignore) 또는 KQ11 fixture 없음")
class TestFullCycleParity:
    """검증된 full-cycle 수치 재현 — Sharpe 0.66 / annual +10.5% / MDD -17%."""

    @staticmethod
    def _run():
        from v4.data import load_panel
        close, dvol = load_panel(CACHE)
        idx = pd.read_parquet(KQ11_FIXTURE)["Close"]
        idx.index = pd.to_datetime(idx.index).normalize()
        return run_backtest(close, dvol, idx, KoreaConfig())

    def test_sharpe_reproduced(self):
        r = self._run()
        assert 0.60 <= r.sharpe <= 0.72, f"Sharpe {r.sharpe:.3f} != ~0.66"

    def test_annual_reproduced(self):
        r = self._run()
        assert 0.085 <= r.annual <= 0.125, f"annual {r.annual:.3f} != ~0.105"

    def test_mdd_reproduced(self):
        r = self._run()
        assert 0.14 <= abs(r.mdd) <= 0.20, f"MDD {r.mdd:.3f} != ~-0.17"

    def test_invariants(self):
        r = self._run()
        assert r.cash_pct > 0.0          # gate 가 일부 기간 현금
        assert all(np.isfinite(x) for x in r.rets)
        assert r.n_rebal > 100           # full-cycle 충분한 rebalance
