"""Phase 0: vol-self trade edge 검증 (V4, 2026-05-28).

가설: VolTransformer vol 예측(IC 0.70)이 옵션 implied vol(VRP 포함)을 이기면,
direction 무관하게 ATM straddle 매수로 수익 가능.

검증 (historical IV 없으니 realized vol proxy):
  straddle PnL% ≈ |forward 5d move%| − implied_breakeven%
  implied_breakeven% = recent_vol(vol_cc_20d) × sqrt(5/252) × VRP_factor
    · VRP_factor=1.0: implied = historical (VRP 무시, 낙관)
    · VRP_factor=1.2~1.3: 옵션 매도자 프리미엄 반영 (현실적)

go/no-go: VolTransformer high vol_score quintile의 평균 straddle PnL이
VRP_factor 1.2에서 양수면 edge 가능성 (단 옵션 bid-ask/commission 추가 차감 필요).
0 근처/음수면 프로젝트 중단.

주의: ATM 만기보유 근사. 옵션 spread(1~5%)/commission 미반영 → 낙관적 추정.
실제 net edge는 이보다 낮음.

Usage:
    PYTHONPATH=. python v3/research/test_vol_straddle_edge.py
"""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.config.schema import load_config
from v3.utils.storage import StorageManager
from v3.utils.logger import setup_logger
from v3.utils.device import set_seed
from v3.scripts.run_backtest import generate_vol_predictions


HORIZON = 5
VRP_FACTORS = [1.0, 1.2, 1.3]


def main() -> int:
    setup_logger()
    set_seed(42)
    cfg = load_config(None)
    storage = StorageManager(base_dir="v3")

    logger.info("Loading processed data...")
    df = storage.load_parquet("processed_data")
    feat_cfg = storage.load_json("feature_config.json")
    feature_cols = feat_cfg["feature_cols"]

    # VolTransformer 예측 (test period)
    vol_preds = generate_vol_predictions(cfg, storage, df, feature_cols)
    logger.info(f"vol predictions: {len(vol_preds)} rows")

    # forward HORIZON-day |move| + recent vol
    d = df.sort_values(["ticker", "date"]).copy()
    d["fwd_close"] = d.groupby("ticker")["close"].shift(-HORIZON)
    d["fwd_move_abs"] = (d["fwd_close"] / d["close"] - 1.0).abs()

    cols = ["date", "ticker", "close", "vol_cc_20d", "fwd_move_abs"]
    panel = vol_preds.merge(d[cols], on=["date", "ticker"], how="inner")
    panel = panel.dropna(subset=["fwd_move_abs", "vol_cc_20d", "vol_score"])
    panel = panel[panel["vol_cc_20d"] > 0]
    logger.info(f"panel after merge/dropna: {len(panel)} rows")

    # implied breakeven + straddle PnL per VRP
    sqrt_t = np.sqrt(HORIZON / 252.0)
    for vrp in VRP_FACTORS:
        be = panel["vol_cc_20d"] * sqrt_t * vrp
        panel[f"pnl_{vrp}"] = panel["fwd_move_abs"] - be

    # vol_score quintile (5 = highest predicted vol expansion)
    panel["vol_q"] = pd.qcut(panel["vol_score"], 5, labels=False, duplicates="drop")

    logger.info("=" * 64)
    logger.info("VOL-SELF TRADE EDGE — straddle PnL by vol_score quintile")
    logger.info(f"panel={len(panel)}, horizon={HORIZON}d (ATM 만기보유 근사)")
    logger.info("=" * 64)
    logger.info("PnL% = |forward move| − recent_vol×sqrt(5/252)×VRP  (옵션 spread/comm 미반영)")
    logger.info("")

    summary = {}
    for vrp in VRP_FACTORS:
        col = f"pnl_{vrp}"
        by_q = panel.groupby("vol_q")[col].agg(["mean", "median", "count"])
        logger.info(f"--- VRP_factor = {vrp} ---")
        for q, row in by_q.iterrows():
            tag = " <- highest predicted vol" if q == by_q.index.max() else ""
            logger.info(
                f"  Q{int(q)}: mean PnL={row['mean']:+.4f}  median={row['median']:+.4f}  "
                f"n={int(row['count'])}{tag}"
            )
        high_q = by_q.index.max()
        low_q = by_q.index.min()
        spread = float(by_q.loc[high_q, "mean"] - by_q.loc[low_q, "mean"])
        logger.info(f"  high−low quintile spread: {spread:+.4f}")
        logger.info("")
        summary[str(vrp)] = {
            "high_q_mean_pnl": float(by_q.loc[high_q, "mean"]),
            "low_q_mean_pnl": float(by_q.loc[low_q, "mean"]),
            "high_low_spread": spread,
            "overall_mean_pnl": float(panel[col].mean()),
        }

    # verdict
    logger.info("=" * 64)
    logger.info("VERDICT (VRP=1.2 현실적 기준):")
    h12 = summary["1.2"]["high_q_mean_pnl"]
    if h12 > 0.01:
        v = "PASS — high vol quintile straddle PnL > +1% (옵션 비용 감안 검토 가치)"
    elif h12 > 0:
        v = "MARGINAL — 양수지만 작음 (옵션 spread/comm에 잠식될 위험)"
    else:
        v = "FAIL — high vol quintile straddle PnL ≤ 0 (VRP 못 이김, 프로젝트 중단)"
    logger.info(f"  high_q PnL (VRP 1.2) = {h12:+.4f} → {v}")
    logger.info("=" * 64)

    out = Path("v3/research/reports/vol_straddle_edge.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon": HORIZON,
        "panel_size": len(panel),
        "vrp_factors": VRP_FACTORS,
        "summary": summary,
        "verdict_vrp12_high_q_pnl": h12,
        "note": "ATM 만기보유 근사, 옵션 bid-ask/commission 미반영 → 낙관적",
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
