"""V4 Korea backtest — 검증 엔진을 패널 위로 돌려 full-cycle 재현.

엔진(engine.py)의 동일 함수(ensemble_picks/regime_on/target_exposure)를 호출 →
백테스트-라이브 단일 코드 경로. 차이는 backtest 가 forward 수익을 알고, live 는 모름.

검증 목표: KOSDAQ 2014-2026 Sharpe ~0.66 / annual +10.5% / MDD -17% 재현.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import KoreaConfig
from .engine import Picks, ensemble_picks, regime_on, target_exposure


@dataclass(frozen=True)
class BacktestResult:
    annual: float
    sharpe: float
    mdd: float
    total_return: float
    win_rate: float
    n_rebal: float
    cash_pct: float            # 노출 0 (gate off) 비율
    rets: tuple[float, ...]


def _position_return(path: pd.Series, entry: float, cfg: KoreaConfig) -> float:
    """1포지션 hold 수익. 만기 종가 / 상폐 시 last-valid − delist_pen / 전손."""
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    lv = path.last_valid_index()
    if lv is not None and path[lv] > 0:
        return path[lv] / entry - 1.0 - cfg.delist_pen
    return -1.0


def _basket_forward_return(close: pd.DataFrame, i: int, picks: Picks, cfg: KoreaConfig) -> float:
    """픽들의 equal-weight hold 수익 − 비용 (raw, gate/노출 적용 전)."""
    if not picks.tickers:
        return 0.0
    pr = [_position_return(close.iloc[i:i + cfg.hold + 1][t], close.iloc[i][t], cfg)
          for t in picks.tickers]
    return float(np.mean(pr)) - cfg.cost


def run_backtest(close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series,
                 cfg: KoreaConfig = KoreaConfig()) -> BacktestResult:
    """엔진을 rebalance loop 로 구동. raw basket(phantom)으로 vol-target conditioning."""
    dates = close.index
    basket_rets: list[float] = []     # raw(phantom) — vol-target history (gate 무관)
    realized: list[float] = []        # 실현 = gate × vol-target × basket
    cash_count = 0
    for i in range(cfg.max_lb, len(dates) - cfg.hold, cfg.hold):
        picks = ensemble_picks(close, dvol, i, cfg)
        raw = _basket_forward_return(close, i, picks, cfg)
        exp = target_exposure(regime_on(index, dates[i], cfg), basket_rets, cfg)
        realized.append(exp * raw)
        basket_rets.append(raw)       # raw 는 노출과 무관하게 history 에 누적
        if exp <= 0:
            cash_count += 1

    r = np.asarray(realized)
    if len(r) == 0 or not np.all(np.isfinite(r)):
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, tuple())
    eq = np.cumprod(1 + r)
    total = float(eq[-1] - 1)
    rpy = cfg.rebal_per_year
    ann = float((1 + total) ** (rpy / len(r)) - 1) if total > -1 else -1.0
    vol = float(r.std() * np.sqrt(rpy))
    sharpe = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return BacktestResult(
        annual=ann, sharpe=sharpe, mdd=mdd, total_return=total,
        win_rate=float((r > 0).mean()), n_rebal=len(r),
        cash_pct=cash_count / len(r), rets=tuple(float(x) for x in r),
    )
