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

    # 1. 직전 phantom 픽 실현 수익 측정 (commit 시점까지 누적 보류 — 실패 시 double-count 방지)
    measured = _measure_pending(state, close, i, cfg)

    # 2. 오늘 신호 (엔진 동일 함수)
    picks = ensemble_picks(close, dvol, i, cfg)        # phantom (gate 무관)
    on = regime_on(index, today, cfg)
    exp = target_exposure(on, state.basket_history, cfg)
    book = ({t: exp / len(picks.tickers) for t in picks.tickers}
            if picks.tickers and exp > 0 else {})

    # 3. 주문 reconcile — sizing은 패널 종가로 (KIS 시세 불요). gate off → book {} → 전량 청산
    prices = close.iloc[i].dropna().to_dict()
    plan = rebalance(broker, book, prices, execute=execute)

    executed = len(plan.sells) + len(plan.buys)
    failed = len(plan.failures)

    # 3b. 전 주문 실행 실패 → state 미전진 (silent failure 방지, 다음 세션 재시도).
    #     2026-06-07 회귀: 6/1 우선주 매매불가+자금부족 전량 실패인데 state 전진 → 빈 거래 방치.
    if failed > 0 and executed == 0:
        logger.error(f"rebalance {today.date()}: 전 주문 실패 ({failed}건) — state 미전진, "
                     f"다음 세션 재시도. (계좌 리셋/매매가능/예수금 점검 필요)")
        return SessionResult(False, plan, exp, on, len(picks.tickers), measured,
                             "exec_failed")

    # 4. commit — phantom 측정 누적 + pending 갱신 + rebalance 완료 기록
    if measured is not None:
        state.basket_history.append(measured)
    state.pending_entries = {t: float(close.iloc[i][t]) for t in picks.tickers}
    state.pending_date = str(today.date())
    state.last_rebalance_date = str(today.date())

    if failed > 0:
        logger.warning(f"rebalance {today.date()}: 부분 실패 — 성공 {executed}, 실패 {failed}. "
                       f"계좌 부분 정렬 상태(다음 주기에 재정렬).")
    logger.info(f"rebalance {today.date()}: regime={'ON' if on else 'OFF'} "
                f"exposure={exp:.2f} picks={len(picks.tickers)} "
                f"sells={len(plan.sells)} buys={len(plan.buys)} fails={failed}")
    return SessionResult(True, plan, exp, on, len(picks.tickers), measured,
                         "rebalanced")
