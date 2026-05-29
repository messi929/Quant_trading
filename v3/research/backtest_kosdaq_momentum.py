"""KOSDAQ TS momentum net 백테스트 (V4, 2026-05-29).

확정된 신호(TS momentum alpha +2.39%/20d gross)를 실제 trend-following
포트폴리오로 시뮬 — 한국 거래비용 반영한 net 수익 측정.

전략:
  - 20 거래일마다 rebalance
  - past 60일 수익률 > 0 (추세) 종목 중 momentum 상위 N개
  - equal weight, 20일 보유 후 재선정
  - 비용: 한국 왕복 (거래세 0.18~0.23% + 수수료 + 슬리피지) → 시나리오 0.3% / 0.5%

survivorship-free: 캐시에 상폐 종목 포함. 상폐 직전 폭락은 past60 음수라
추세 진입 자체가 안 됨 (자연 회피). fwd NaN 종목은 제외.

benchmark: KOSDAQ 지수 (KQ11).
PASS: net annual return > ~10% AND net Sharpe > 0.7 (무위험+α, 운영 가치).

Usage:
    PYTHONPATH=. python v3/research/backtest_kosdaq_momentum.py
"""

from __future__ import annotations

import sys, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import warnings; warnings.filterwarnings("ignore")

import FinanceDataReader as fdr

LOOKBACK = 60
HOLD = 20
CACHE = Path("v3/research/reports/korea_kosdaq_ohlcv_cache.parquet")
REBAL_PER_YEAR = 252 / HOLD   # ~12.6


def backtest(piv: pd.DataFrame, n_pos: int, cost_roundtrip: float,
             delisting_penalty: float = 0.5) -> dict:
    """20일 rebalance trend-following. 상폐 손실 반영 (survivorship-free 진짜).

    픽 선정: past60>0 momentum 상위 N (fwd 조건 제거 → 상폐 예정 종목도 포함).
    상폐 처리: 보유 중 데이터 끊기면 마지막 유효가까지 수익 − delisting_penalty
    (한국 정리매매 손실). 진입 직후 상폐면 전손(-100%).
    """
    dates = piv.index.tolist()
    rets = []
    prev_holdings = set()
    total_picks, total_delisted = 0, 0
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        past = (piv.iloc[i] / piv.iloc[i - LOOKBACK] - 1.0)
        past = past[past.notna()]          # fwd 조건 제거 (상폐 예정 포함)
        trend = past[past > 0]
        if len(trend) == 0:
            rets.append(0.0)
            prev_holdings = set()
            continue
        picks = trend.nlargest(min(n_pos, len(trend))).index
        pick_rets = []
        for t in picks:
            entry = piv.iloc[i][t]
            exit_p = piv.iloc[i + HOLD][t]
            if pd.notna(exit_p):
                r = exit_p / entry - 1.0
            else:
                # 보유 중 상폐 — 마지막 유효가까지 + 정리매매 손실
                window = piv.iloc[i:i + HOLD + 1][t]
                lv = window.last_valid_index()
                if lv is not None and pd.notna(window[lv]) and window[lv] > 0:
                    r = (window[lv] / entry - 1.0) - delisting_penalty
                else:
                    r = -1.0
                total_delisted += 1
            pick_rets.append(r)
        total_picks += len(picks)
        gross = float(np.mean(pick_rets))
        new = set(picks) - prev_holdings
        turnover = len(new) / max(len(picks), 1)
        cost = cost_roundtrip * turnover
        rets.append(gross - cost)
        prev_holdings = set(picks)

    rets = np.array(rets)
    if len(rets) == 0:
        return {}
    eq = np.cumprod(1 + rets)
    total_ret = float(eq[-1] - 1)
    ann_ret = float((1 + total_ret) ** (REBAL_PER_YEAR / len(rets)) - 1)
    vol = float(rets.std() * np.sqrt(REBAL_PER_YEAR))
    sharpe = float(ann_ret / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    win = float((rets > 0).mean())
    return {
        "n_pos": n_pos, "cost": cost_roundtrip, "delisting_penalty": delisting_penalty,
        "total_return": total_ret, "annual_return": ann_ret,
        "sharpe": sharpe, "mdd": mdd, "win_rate": win,
        "n_rebal": len(rets), "avg_ret_per_rebal": float(rets.mean()),
        "delisted_picks": total_delisted, "total_picks": total_picks,
        "delisted_pct": float(total_delisted / max(total_picks, 1)),
    }


def main() -> int:
    if not CACHE.exists():
        logger.error(f"cache 없음: {CACHE} — test_korea_ts_alpha_final.py 먼저 실행")
        return 1
    panel = pd.read_parquet(CACHE)
    piv = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    logger.info(f"universe: {piv.shape[1]} tickers, {piv.shape[0]} days")

    # benchmark 지수
    idx = fdr.DataReader("KQ11", "2021-01-01", "2026-05-01")["Close"]
    idx_ret = idx.iloc[-1] / idx.iloc[0] - 1
    days = (idx.index[-1] - idx.index[0]).days
    idx_ann = (1 + idx_ret) ** (365 / days) - 1
    logger.info(f"KOSDAQ 지수 (benchmark): total {idx_ret:+.1%}, annual {idx_ann:+.1%}")

    logger.info("=" * 70)
    logger.info("KOSDAQ TS momentum NET backtest (상폐 손실 반영, 20d rebal)")
    logger.info("=" * 70)
    results = []
    for dp in [0.3, 0.5]:
        logger.info(f"--- 상폐 정리매매 손실 penalty = {dp:.0%} ---")
        for n in [10, 20, 30]:
            r = backtest(piv, n, 0.004, delisting_penalty=dp)
            results.append(r)
            logger.info(
                f"  N={n:2d}: annual={r['annual_return']:+.1%}  Sharpe={r['sharpe']:+.2f}  "
                f"MDD={r['mdd']:.1%}  win={r['win_rate']:.0%}  "
                f"상폐픽={r['delisted_pct']:.1%}({r['delisted_picks']})"
            )
        logger.info("")
    # 대표 (cost 0.4%, N=20, penalty 0.5)
    best = backtest(piv, 20, 0.004, delisting_penalty=0.5)
    logger.info(f"대표 (cost 0.4%, N=20, 상폐penalty 50%): annual {best['annual_return']:+.1%}, "
                f"Sharpe {best['sharpe']:.2f}, MDD {best['mdd']:.1%}, 상폐픽 {best['delisted_pct']:.1%}")
    logger.info("")
    logger.info("VERDICT (cost 0.4%, N=20, 상폐 50% penalty 기준):")
    if best["annual_return"] > 0.10 and best["sharpe"] > 0.7:
        logger.info(f"  PASS — net annual {best['annual_return']:+.1%} > 10%, Sharpe {best['sharpe']:.2f} > 0.7.")
        logger.info(f"    무위험+α 운영 가치. 시스템 설계 진행.")
    elif best["annual_return"] > 0.05:
        logger.info(f"  MARGINAL — net annual {best['annual_return']:+.1%} (무위험 근처). 최적화 필요.")
    else:
        logger.info(f"  FAIL — net annual {best['annual_return']:+.1%}. 비용이 alpha 잠식.")
    logger.info("=" * 70)

    out = Path("v3/research/reports/kosdaq_momentum_backtest.json")
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_tickers": int(piv.shape[1]),
        "kosdaq_index_annual": float(idx_ann),
        "grid": results,
        "representative_cost004_n20": best,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
