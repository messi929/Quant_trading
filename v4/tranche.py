"""V4 트렌치 분할 백테스트 — window luck 거세 (증분 1, 2026-06-19).

배경: 단일 phase(rebalance offset 0) 백테스트는 '어느 날 리밸런스했는가'(window luck)에
종속 — 20 phase 측정 시 Sharpe 0.09~0.69(스프레드 0.60), MDD −19~−49%. 보고돼온
offset=0(Sharpe~0.66/MDD−17%)은 최상위 phase = luck. 정직한 phase 평균 ≈ 0.47/−35%.
자본을 N 트렌치로 쪼개 시차 순환하면 진입·청산 시점이 평활화 → robust ~0.56/−20%.

여기선 engine.py 동일 함수(ensemble_picks/regime_on/target_exposure)로 production 재현 —
N개 phase 일일-MTM 1/N 평균. N=1 이면 단일 phase(일일-MTM). research(korea_tranche.py)와
동일 방법론, 이제 엔진 코드 경로 위에서. 라이브 무위험 — 백테스트 전용.

증분 2: live runner 가 이 결합 book(Σ tranche 목표) 을 매일 산출하도록 mirror (parity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import _basket_forward_return
from .config import KoreaConfig
from .engine import ensemble_picks, regime_on, target_exposure

ANN = 252.0


@dataclass(frozen=True)
class TrancheResult:
    n_tranches: int
    offsets: tuple[int, ...]
    sharpe: float
    annual: float
    mdd: float
    total_return: float
    phase_sharpes: tuple[float, ...]     # 개별 phase 단일운용 (참고 — window luck 분산)
    phase_mdds: tuple[float, ...]


def tranche_offsets(n_tranches: int, hold: int) -> list[int]:
    """N 트렌치 균등 시차 offset (hold 주기를 N등분). N=5, hold=20 → [0,4,8,12,16]."""
    return [int(round(k * hold / n_tranches)) % hold for k in range(n_tranches)]


def phase_daily_returns(close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series,
                        offset: int, cfg: KoreaConfig = KoreaConfig()) -> pd.Series:
    """단일 phase(시작 offset)의 일일 수익 Series. engine 동일 함수 + 일일 MTM.

    각 rebalance: phantom 픽(gate 무관)으로 vol-target history 누적(backtest parity),
    gate×vol-target 노출로 보유창 일일 등가중 수익. gate off/픽 없음 → 그 창은 현금(0).
    """
    dates = close.index
    ridx = [i for i in range(cfg.max_lb + offset, len(dates) - cfg.hold, cfg.hold)]
    basket_hist: list[float] = []
    daily = pd.Series(0.0, index=dates)
    for i in ridx:
        picks = ensemble_picks(close, dvol, i, cfg)
        raw = _basket_forward_return(close, i, picks, cfg)        # phantom (vol-target용)
        exp = target_exposure(regime_on(index, dates[i], cfg), basket_hist, cfg)
        basket_hist.append(raw)
        if exp <= 0 or not picks.tickers:
            continue
        win = close.iloc[i:i + cfg.hold + 1][list(picks.tickers)]
        dret = win.ffill().pct_change(fill_method=None).iloc[1:].mean(axis=1).fillna(0.0)  # 보유창 일일 등가중
        seg = (exp * dret).copy()
        seg.iloc[0] = seg.iloc[0] - cfg.cost                      # 진입일 왕복비용
        daily.loc[seg.index] = seg.values
    return daily


def _daily_stats(r: pd.Series) -> tuple[float, float, float, float]:
    """일일 수익 → (sharpe, annual, mdd, total). 첫 비현금일부터 계산."""
    rv = np.asarray(r.values, dtype=float)
    nz = np.flatnonzero(rv != 0)
    if len(nz) == 0:
        return 0.0, 0.0, 0.0, 0.0
    rv = rv[nz[0]:]
    if rv.std() < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    eq = np.cumprod(1 + rv); total = float(eq[-1] - 1)
    yrs = len(rv) / ANN
    ann = float((1 + total) ** (1 / yrs) - 1) if (total > -1 and yrs > 0) else -1.0
    sharpe = float(rv.mean() / rv.std() * np.sqrt(ANN))
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return sharpe, ann, mdd, total


def run_tranche_backtest(close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series,
                         cfg: KoreaConfig = KoreaConfig(), n_tranches: int = 5) -> TrancheResult:
    """N 트렌치 균등 시차 일일-MTM 블렌드. N=1 → 단일 phase. window luck 거세 측정."""
    offs = tranche_offsets(n_tranches, cfg.hold)
    phases = [phase_daily_returns(close, dvol, index, o, cfg) for o in offs]
    pstats = [_daily_stats(p) for p in phases]
    blend = sum(phases) / n_tranches              # 1/N 자본 시차 결합 (일일 MTM)
    sh, ann, mdd, total = _daily_stats(blend)
    return TrancheResult(
        n_tranches=n_tranches, offsets=tuple(offs),
        sharpe=sh, annual=ann, mdd=mdd, total_return=total,
        phase_sharpes=tuple(s[0] for s in pstats),
        phase_mdds=tuple(s[2] for s in pstats),
    )
