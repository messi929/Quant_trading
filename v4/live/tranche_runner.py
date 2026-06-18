"""V4 트렌치 live runner — N 하부포트 시차 운용 (증분 2, 2026-06-19).

window luck 거세(v4/tranche.py 검증: N=5 Sharpe 0.56/MDD −20% vs 단일 phase 0.47/−35%)
를 라이브에 반영. 단일 LiveState/run_session 의 트렌치 버전.

매 세션:
  1. anchor(첫 세션일) 기준 due 트렌치 판정:
     - 첫 진입 전: trading_days_since(anchor) >= t·step (시차 부트스트랩, 20일에 걸쳐 1/N→full)
     - 이후: 각 트렌치 자체 20거래일 주기.
  2. due 트렌치만 엔진 동일 함수로 새 book(노출×1/N) 계산.
  3. 결합 book = Σ 모든 트렌치 target_weights(due=신규, 그 외=기존) → executor reconcile.
  4. 전 주문 실패 시 due 트렌치 state 미전진(silent failure 방지). 부분/성공 시 commit.

backtest parity: 각 트렌치가 run_session 과 동일 ensemble_picks/regime_on/target_exposure
호출 → 결합이 run_tranche_backtest 와 동일 구조(단일 코드 경로).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from v4.config import KoreaConfig
from v4.engine import ensemble_picks, regime_on, target_exposure
from v4.execution.executor import RebalancePlan, rebalance
from v4.live.runner import trading_days_since
from v4.live.state import TrancheLiveState, TrancheSlot


@dataclass(frozen=True)
class TrancheSessionResult:
    rebalanced_tranches: tuple[int, ...]   # 이번 세션 리밸런스된 트렌치 인덱스
    plan: RebalancePlan | None
    gross_exposure: float                  # 결합 목표 노출 (Σ weights)
    n_positions: int                       # 결합 book 고유 종목 수
    note: str


def _measure_slot(slot: TrancheSlot, close: pd.DataFrame, i: int, cfg: KoreaConfig) -> float | None:
    """트렌치 직전 phantom 픽 실현 수익 (현재가/entry−1 평균 − 비용). 없으면 None."""
    if not slot.pending_entries:
        return None
    rets = []
    for t, entry in slot.pending_entries.items():
        if entry <= 0:
            continue
        cur = close.iloc[i][t] if t in close.columns else np.nan
        rets.append(cur / entry - 1.0 if pd.notna(cur) else -1.0 - cfg.delist_pen)
    return float(np.mean(rets)) - cfg.cost if rets else None


def _tranche_book(close, dvol, index, i, slot, cfg, n_tranches):
    """due 트렌치의 새 목표 book(노출×1/N) + phantom 픽 entry. engine 동일 함수."""
    today = close.index[i]
    picks = ensemble_picks(close, dvol, i, cfg)                    # phantom (gate 무관)
    exp = target_exposure(regime_on(index, today, cfg), slot.basket_history, cfg)
    if not picks.tickers or exp <= 0:
        book = {}
    else:
        w = (exp / n_tranches) / len(picks.tickers)               # 1/N 자본 등가중
        book = {t: w for t in picks.tickers}
    entries = {t: float(close.iloc[i][t]) for t in picks.tickers}
    return book, entries, picks.tickers


def _due_tranches(state: TrancheLiveState, index: pd.Series, today: pd.Timestamp,
                  cfg: KoreaConfig) -> list[int]:
    step = max(1, cfg.hold // state.n_tranches)
    due = []
    for t, slot in enumerate(state.tranches):
        if slot.last_rebalance_date is None:                       # 첫 진입 (시차 부트스트랩)
            elapsed = trading_days_since(index, state.anchor_date, today) if state.anchor_date else 0
            if elapsed >= t * step:
                due.append(t)
        elif trading_days_since(index, slot.last_rebalance_date, today) >= cfg.hold:
            due.append(t)
    return due


def run_tranche_session(broker, close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series,
                        state: TrancheLiveState, cfg: KoreaConfig = KoreaConfig(), *,
                        execute: bool = True) -> TrancheSessionResult:
    """1세션. close/dvol/index 는 today(마지막 행)까지 패널. state in-place 갱신."""
    i = len(close) - 1
    today = close.index[i]
    if state.anchor_date is None:
        state.anchor_date = str(today.date())

    due = _due_tranches(state, index, today, cfg)
    if not due:
        logger.info(f"hold: due 트렌치 없음 (활성 {sum(s.last_rebalance_date is not None for s in state.tranches)}"
                    f"/{state.n_tranches})")
        # 결합 book 유지 — 무행동
        return TrancheSessionResult((), None, 0.0, 0, "hold")

    # 1. due 트렌치 새 book 계산 (commit 보류 — 실패 시 미전진)
    new_books, new_entries, new_picks, measured = {}, {}, {}, {}
    for t in due:
        slot = state.tranches[t]
        measured[t] = _measure_slot(slot, close, i, cfg)
        new_books[t], new_entries[t], new_picks[t] = _tranche_book(
            close, dvol, index, i, slot, cfg, state.n_tranches)

    # 2. 결합 book = Σ (due=신규, 그 외=기존)
    combined: dict[str, float] = {}
    for t, slot in enumerate(state.tranches):
        wts = new_books[t] if t in new_books else slot.target_weights
        for tk, w in wts.items():
            combined[tk] = combined.get(tk, 0.0) + w

    # 3. reconcile (sizing 패널 종가, gate off 트렌치 → 빈 기여)
    prices = close.iloc[i].dropna().to_dict()
    plan = rebalance(broker, combined, prices, execute=execute)
    executed = len(plan.sells) + len(plan.buys)
    failed = len(plan.failures)

    # 3b. 전 주문 실패 → due 트렌치 미전진 (다음 세션 재시도)
    if failed > 0 and executed == 0:
        logger.error(f"tranche rebalance {today.date()}: 전 주문 실패 ({failed}건) — "
                     f"트렌치 {due} state 미전진, 다음 세션 재시도.")
        return TrancheSessionResult((), plan, 0.0, len(combined), "exec_failed")

    # 4. commit due 트렌치
    for t in due:
        slot = state.tranches[t]
        if measured[t] is not None:
            slot.basket_history.append(measured[t])
        slot.pending_entries = new_entries[t]
        slot.target_weights = new_books[t]
        slot.last_rebalance_date = str(today.date())

    gross = sum(combined.values())
    if failed > 0:
        logger.warning(f"tranche rebalance {today.date()}: 부분 실패 — 성공 {executed}, 실패 {failed}.")
    logger.info(f"tranche rebalance {today.date()}: 트렌치 {due} 리밸런스, "
                f"결합노출={gross:.2f} 종목={len(combined)} "
                f"sells={len(plan.sells)} buys={len(plan.buys)} fails={failed}")
    return TrancheSessionResult(tuple(due), plan, gross, len(combined),
                                "rebalanced")
