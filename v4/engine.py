"""V4 Korea engine 코어 — backtest·live 공용 순수 함수 (단일 코드 경로).

CLAUDE.md 정합성 원칙: 백테스트와 라이브가 동일 함수를 호출. 차이는 데이터 공급뿐.

핵심 분리:
  - ensemble_picks(): gate 무관 momentum 종목 선정 (PIT universe + 4-lb 랭크평균 + TS trend)
  - regime_on():       지수 200d SMA gate (노출 on/off)
  - vol_target_exposure(): 전략 trailing 변동성으로 노출 역가중
  - target_book():     위를 합쳐 ticker→weight (live 진입점)

vol-target 은 raw(phantom) basket 수익 history 로 conditioning (검증 parity). live 는
과거 픽들의 실현 20d 수익을 rolling 유지해 동일 history 구성.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import KoreaConfig


@dataclass(frozen=True)
class Picks:
    """rebalance 시점 종목 선정 결과 (gate/노출 적용 전, immutable)."""
    date: pd.Timestamp
    tickers: tuple[str, ...]
    n_candidates: int


def ensemble_picks(close: pd.DataFrame, dvol: pd.DataFrame, i: int, cfg: KoreaConfig) -> Picks:
    """i 시점 momentum 픽 (gate 무관). PIT 거래대금 pool → 4-lb 랭크평균 → TS trend → 상위 N."""
    d = close.index[i]
    liq = dvol.iloc[i].dropna()
    if len(liq) < cfg.min_candidates:
        return Picks(d, (), 0)
    pool = liq.nlargest(cfg.liq_top).index
    pasts = {lb: close.iloc[i][pool] / close.iloc[i - lb][pool] - 1.0 for lb in cfg.lookbacks}
    df = pd.DataFrame(pasts).dropna()                  # 4개 lb 모두 유효한 종목
    if len(df) < cfg.min_candidates:
        return Picks(d, (), len(df))
    mean_past = df.mean(axis=1)                         # TS trend filter
    blended = df.rank().mean(axis=1)                    # cross-sectional 랭크 평균
    cand = blended[mean_past > 0]                       # 추세(양수) 종목만
    if len(cand) == 0:
        return Picks(d, (), 0)
    picks = tuple(cand.nlargest(min(cfg.n_pos, len(cand))).index)
    return Picks(d, picks, len(cand))


def regime_on(index: pd.Series, as_of: pd.Timestamp, cfg: KoreaConfig) -> bool:
    """지수 > 200d SMA 면 risk-on. causal (as_of 이전 데이터만). NaN → off."""
    sma = index.rolling(cfg.sma_window, min_periods=cfg.sma_window // 2).mean()
    iv = index.asof(as_of)
    sv = sma.asof(as_of)
    return bool(pd.notna(iv) and pd.notna(sv) and iv > sv)


def vol_target_exposure(recent_basket_rets: Sequence[float], cfg: KoreaConfig) -> float:
    """전략 trailing 변동성으로 노출 역가중. history < vol_win → 1.0 (scaling 보류)."""
    if len(recent_basket_rets) < cfg.vol_win:
        return 1.0
    rv = float(np.std(np.asarray(recent_basket_rets[-cfg.vol_win:])))
    if rv <= 1e-9:
        return cfg.vol_cap
    tgt = cfg.target_vol / np.sqrt(cfg.rebal_per_year)
    return float(np.clip(tgt / rv, 0.0, cfg.vol_cap))


def target_exposure(is_regime_on: bool, recent_basket_rets: Sequence[float], cfg: KoreaConfig) -> float:
    """총 노출 = gate(0/1) × vol-target. gate off 면 0 (전량 현금)."""
    if not is_regime_on:
        return 0.0
    return vol_target_exposure(recent_basket_rets, cfg)


def target_book(close: pd.DataFrame, dvol: pd.DataFrame, index: pd.Series, i: int,
                recent_basket_rets: Sequence[float], cfg: KoreaConfig) -> dict[str, float]:
    """live 진입점 — i 시점 목표 포트폴리오 ticker→weight. 미배치분은 현금(합<1).

    weight = exposure / len(picks) (검증 parity: 픽 equal-weight, 총 invested=exposure).
    """
    picks = ensemble_picks(close, dvol, i, cfg)
    exp = target_exposure(regime_on(index, close.index[i], cfg), recent_basket_rets, cfg)
    if not picks.tickers or exp <= 0:
        return {}
    w = exp / len(picks.tickers)
    return {t: w for t in picks.tickers}
