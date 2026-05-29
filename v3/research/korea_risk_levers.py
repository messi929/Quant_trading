"""한국 vol-targeting + trailing stop lever 테스트 (V4, 2026-05-30).

긴 history(2014-2026) full-cycle: momentum+gate = Sharpe~0.4/MDD~40% (목표 미달).
주범 = momentum crash. 교과서적 해법 vol-targeting(Barroso & Santa-Clara) +
trailing stop 추가해 full-cycle Sharpe 끌어올리는지 + COVID whipsaw 개선되는지.

lever:
  1. vol-target : 전략 자체 trailing 변동성으로 노출 역가중. exposure =
     clip(target_vol / realized_vol, 0, cap). 고변동(crash 직후)에 자동 축소.
     게이트(binary)와 달리 연속 반응 → whipsaw 완화 기대.
  2. trailing stop : 보유 중 일별 경로에서 peak 대비 stop% 하락 시 청산
     (momentum 종목 반전 시 손실 제한). 일봉 캐시 활용.

basket return 1회 계산(stop 유무) → gate/voltarget overlay 곱셈. 긴 캐시 재사용.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/korea_risk_levers.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr

LOOKBACK, HOLD, LIQ_TOP, N_POS = 60, 20, 100, 20
COST, DELIST_PEN = 0.004, 0.5
DATA_START, END = "2014-01-01", "2026-05-01"
RPY = 252 / HOLD
TARGET_VOL = 0.15          # 연 변동성 타겟
VOL_WIN = 6                # trailing rebalance 수 (~6개월)
REPORTS = Path("v3/research/reports")
MARKETS = {"KOSDAQ": ("korea_kosdaq_long_cache.parquet", "KQ11"),
           "KOSPI": ("korea_kospi_long_cache.parquet", "KS11")}
CRASHES = {"2018 Q4": ("2018-06-01", "2019-01-31"),
           "2020 COVID": ("2020-01-01", "2020-05-31"),
           "2022 약세장": ("2022-01-01", "2022-12-31")}


def position_return(path: pd.Series, entry: float, stop: float | None) -> float:
    """일별 경로로 1포지션 수익률. stop=peak 대비 trailing %. 상폐=last valid - pen."""
    peak = entry
    last_valid = entry
    for p in path.iloc[1:]:
        if pd.isna(p):
            continue
        last_valid = p
        peak = max(peak, p)
        if stop is not None and p <= peak * (1 - stop):
            return p / entry - 1.0
    # stop 미발동 — 만기 종가 or 상폐 처리
    final = path.iloc[-1]
    if pd.notna(final):
        return final / entry - 1.0
    if last_valid > 0:
        return last_valid / entry - 1.0 - DELIST_PEN
    return -1.0


def basket_returns(close, dvol, stop: float | None) -> pd.Series:
    """raw momentum basket 수익 (gate/voltarget 미적용). stop 옵션 일별 적용."""
    dates = close.index.tolist(); out = {}
    for i in range(LOOKBACK, len(dates) - HOLD, HOLD):
        liq = dvol.iloc[i].dropna()
        if len(liq) < 20:
            out[dates[i]] = 0.0; continue
        pool = liq.nlargest(LIQ_TOP).index
        past = (close.iloc[i][pool] / close.iloc[i - LOOKBACK][pool] - 1.0).dropna()
        trend = past[past > 0]
        if len(trend) == 0:
            out[dates[i]] = 0.0; continue
        picks = trend.nlargest(min(N_POS, len(trend))).index
        pr = [position_return(close.iloc[i:i + HOLD + 1][t], close.iloc[i][t], stop) for t in picks]
        out[dates[i]] = float(np.mean(pr)) - COST
    return pd.Series(out).sort_index()


def gate_series(index, kind, param):
    if kind == "none":
        return pd.Series(True, index=index.index)
    if kind == "sma":
        return index > index.rolling(param, min_periods=param // 2).mean()
    if kind == "mom":
        return (index / index.shift(param) - 1.0) > 0


def overlay(basket: pd.Series, gate: pd.Series | None, vol_target: bool, cap: float = 1.0) -> pd.Series:
    """basket 에 gate(binary) + vol-target(연속) 노출 곱. 미배치분 현금(0)."""
    tgt_per = TARGET_VOL / np.sqrt(RPY)
    rets = []
    vals = basket.values
    for k, (d, b) in enumerate(basket.items()):
        exp = 1.0
        if gate is not None:
            g = gate.asof(d)
            exp *= 1.0 if (g is True or g == True) else 0.0
        if vol_target and k >= VOL_WIN:
            rv = np.std(vals[k - VOL_WIN:k])
            exp *= float(np.clip(tgt_per / rv, 0, cap)) if rv > 1e-9 else cap
        rets.append(exp * b)
    return pd.Series(rets, index=basket.index)


def stat(r):
    r = np.asarray(r)
    if len(r) == 0:
        return {"annual": 0, "sharpe": 0, "mdd": 0, "ret": 0, "n": 0}
    eq = np.cumprod(1 + r); tot = float(eq[-1] - 1)
    ann = float((1 + tot) ** (RPY / len(r)) - 1) if tot > -1 else -1.0
    vol = float(r.std() * np.sqrt(RPY)); sh = float(ann / vol) if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq); mdd = float(((eq - peak) / peak).min())
    return {"annual": ann, "sharpe": sh, "mdd": mdd, "ret": tot, "n": len(r)}


def sub(s: pd.Series, ps, pe):
    return stat(s[(s.index >= ps) & (s.index <= pe)].values)


def main() -> int:
    out_all = {}
    for m, (cfile, icode) in MARKETS.items():
        panel = pd.read_parquet(REPORTS / cfile)
        close = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
        dvol = close * panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()
        index = fdr.DataReader(icode, DATA_START, END)["Close"]; index.index = pd.to_datetime(index.index).normalize()
        gate = gate_series(index, "sma", 200)

        b_nostop = basket_returns(close, dvol, None)
        b_stop = basket_returns(close, dvol, 0.20)

        configs = {
            "gate only": overlay(b_nostop, gate, False),
            "voltarget only": overlay(b_nostop, None, True),
            "gate+voltarget": overlay(b_nostop, gate, True),
            "gate+trailing": overlay(b_stop, gate, False),
            "gate+voltarget+trailing": overlay(b_stop, gate, True),
        }
        logger.info("=" * 82)
        logger.info(f"{m} — risk levers (full-cycle 2014-2026, baseline gate-only Sharpe~0.4)")
        logger.info("=" * 82)
        res = {}
        for nm, s in configs.items():
            st = stat(s.values)
            res[nm] = {"full": st, "crashes": {c: sub(s, ps, pe) for c, (ps, pe) in CRASHES.items()}}
            cov = res[nm]["crashes"]["2020 COVID"]
            logger.info(f"  {nm:26s}: annual={st['annual']:+.1%}  Sharpe={st['sharpe']:+.2f}  "
                        f"MDD={st['mdd']:.1%}  | COVID MDD={cov['mdd']:+.0%} ret={cov['ret']:+.0%}")
        out_all[m] = res
        logger.info("")

    logger.info("VERDICT (full-cycle Sharpe 목표 0.7+):")
    for m in MARKETS:
        best = max(out_all[m].items(), key=lambda kv: kv[1]["full"]["sharpe"])
        b = best[1]["full"]
        logger.info(f"  {m}: best='{best[0]}' Sharpe={b['sharpe']:.2f} annual={b['annual']:+.1%} MDD={b['mdd']:.1%}")
    logger.info("=" * 82)

    p = REPORTS / "korea_risk_levers.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                             "target_vol": TARGET_VOL, "vol_win": VOL_WIN, "markets": out_all},
                            indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
