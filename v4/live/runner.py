"""V4 일일 runner — rebalance-or-hold 결정 + 주문.

momentum 20일 보유 → 매일 거래 안 함. 매 세션:
  1. rebalance일(직전 rebalance 후 hold 거래일 경과 or 첫 실행)인가?
  2. 아니면 hold (무행동).
  3. 맞으면: 직전 phantom 픽의 실현 수익 측정 → basket_history 누적 →
     오늘 ensemble 픽 + regime gate + vol-target → book → executor reconcile →
     state 갱신(오늘 phantom 픽을 pending 으로).

엔진 동일 함수(ensemble_picks/regime_on/target_exposure) 호출 → backtest parity.
data/state 를 인자로 받는 순수에 가까운 구조 → synthetic 테스트 가능.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from v4.config import KoreaConfig
from v4.engine import ensemble_picks, regime_on, target_exposure
from v4.execution.executor import RebalancePlan, rebalance
from v4.live.state import LiveState


@dataclass(frozen=True)
class SessionResult:
    rebalanced: bool
    plan: RebalancePlan | None
    exposure: float
    regime_on: bool
    n_picks: int
    measured_basket_ret: float | None
    note: str


def trading_days_since(index: pd.Series, since_date: str, today: pd.Timestamp) -> int:
    """index(거래일) 기준 since_date 초과 ~ today 이하 거래일 수."""
    d0 = pd.Timestamp(since_date)
    days = index.index[(index.index > d0) & (index.index <= today)]
    return len(days)


def is_rebalance_day(state: LiveState, index: pd.Series, today: pd.Timestamp, cfg: KoreaConfig) -> bool:
    if state.last_rebalance_date is None:
        return True
    return trading_days_since(index, state.last_rebalance_date, today) >= cfg.hold


def _measure_pending(state: LiveState, close: pd.DataFrame, i: int, cfg: KoreaConfig) -> float | None:
    """직전 phantom 픽의 실현 수익 (현재가/entry−1 평균 − 비용). 없으면 None."""
    if not state.pending_entries:
        return None
    rets = []
    for t, entry in state.pending_entries.items():
        if entry <= 0:
            continue
        cur = close.iloc[i][t] if t in close.columns else np.nan
        if pd.notna(cur):
            rets.append(cur / entry - 1.0)
        else:                                   # 상장폐지/데이터 소실 → 보수적 손실
            rets.append(-1.0 - cfg.delist_pen)
    if not rets:
        return None
    return float(np.mean(rets)) - cfg.cost


def run_session(broker, close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series,
                state: LiveState, cfg: KoreaConfig = KoreaConfig(), *,
                execute: bool = True) -> SessionResult:
    """1세션 실행. close/dvol/index 는 today(마지막 행)까지의 패널. state 는 in-place 갱신."""
    i = len(close) - 1
    today = close.index[i]

    if not is_rebalance_day(state, index, today, cfg):
        elapsed = trading_days_since(index, state.last_rebalance_date, today) if state.last_rebalance_date else 0
        logger.info(f"hold: rebalance까지 {cfg.hold - elapsed}거래일 남음")
        return SessionResult(False, None, 0.0, False, 0, None, "hold")

    # 1. 직전 phantom 픽 실현 수익 측정 → vol-target history 누적
    measured = _measure_pending(state, close, i, cfg)
    if measured is not None:
        state.basket_history.append(measured)

    # 2. 오늘 신호 (엔진 동일 함수)
    picks = ensemble_picks(close, dvol, i, cfg)        # phantom (gate 무관)
    on = regime_on(index, today, cfg)
    exp = target_exposure(on, state.basket_history, cfg)
    book = ({t: exp / len(picks.tickers) for t in picks.tickers}
            if picks.tickers and exp > 0 else {})

    # 3. 주문 reconcile (gate off → book {} → 전량 청산)
    plan = rebalance(broker, book, execute=execute)

    # 4. state 갱신 — 오늘 phantom 픽을 pending 으로 (다음 rebalance에 측정)
    state.pending_entries = {t: float(close.iloc[i][t]) for t in picks.tickers}
    state.pending_date = str(today.date())
    state.last_rebalance_date = str(today.date())

    logger.info(f"rebalance {today.date()}: regime={'ON' if on else 'OFF'} "
                f"exposure={exp:.2f} picks={len(picks.tickers)} "
                f"sells={len(plan.sells)} buys={len(plan.buys)}")
    return SessionResult(True, plan, exp, on, len(picks.tickers), measured,
                         "rebalanced")
