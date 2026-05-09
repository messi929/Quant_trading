"""Prepare 3 input parquets for run_calibration_pipeline.py.

Builds:
  data/research/ohlcv_panel.parquet      — long-format OHLCV passthrough
  data/research/macro_pctl.parquet       — rolling 5y macro percentiles
  data/research/vol_predictions.parquet  — per-(date,ticker) vol inference
                                           over the full historical window

vol_predictions is the slow step: VolTransformer forward per (date, ticker).
Per-date batching keeps it ~20-30 min on server CPU for 5y × 99 tickers.

Usage (server, one-shot before first calibration):
  PYTHONPATH=/opt/quant /opt/quant/venv/bin/python \\
      v3/scripts/prep_calibration_inputs.py

Required inputs on server (already present after deploy):
  v3/data/raw/ohlcv_raw.parquet
  v3/data/raw/macro.parquet
  v3/saved_models/vol_transformer_*.pt
  v3/saved_models/normalizer_stats.json
  v3/saved_models/feature_config.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from v3.config.schema import load_config
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.macro_features import MacroFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.model.vol_transformer import VolTransformer
from v3.utils.device import DeviceManager
from v3.utils.logger import setup_logger
from v3.utils.storage import StorageManager


def build_macro_pctl(macro: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    mfe = MacroFeatureEngineer()
    macro_feats = mfe.compute(macro, ohlcv=ohlcv)
    macro_pctl = mfe.compute_percentiles(macro_feats)
    return macro_pctl


def build_vol_predictions(
    df_feats: pd.DataFrame,
    feature_cols: list[str],
    cfg,
    storage: StorageManager,
) -> pd.DataFrame:
    """Per-date batched inference. Returns long-format predictions."""
    normalizer = FeatureNormalizer.load(cfg.paths.normalizer_stats)
    df_norm = normalizer.transform(df_feats, feature_cols)

    dm = DeviceManager(compile_model=False)
    checkpoint = storage.load_checkpoint("vol_transformer")
    model = VolTransformer(
        input_dim=len(feature_cols),
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_length=cfg.model.max_seq_length,
        use_confidence_head=cfg.model.use_confidence_head,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(dm.device).eval()

    seq_len = cfg.data.sequence_length
    df_norm = df_norm.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_norm["date"] = pd.to_datetime(df_norm["date"])

    # Pre-build per-ticker arrays for fast slicing.
    # dates as int64 nanoseconds → numpy searchsorted works regardless of
    # whether `t` is np.datetime64 or pd.Timestamp.
    ticker_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, group in df_norm.groupby("ticker"):
        dates_ns = group["date"].astype("int64").to_numpy()
        feats = np.nan_to_num(
            group[feature_cols].to_numpy(), nan=0.0,
        ).astype(np.float32)
        ticker_arrays[ticker] = (dates_ns, feats)

    all_dates = sorted(pd.to_datetime(df_norm["date"]).unique())
    logger.info(
        f"Inference: {len(all_dates)} dates × {len(ticker_arrays)} tickers"
    )

    predictions: list[dict] = []
    with torch.no_grad():
        for i, t in enumerate(all_dates):
            t_ns = pd.Timestamp(t).value  # int64 ns
            if i % 100 == 0:
                logger.info(f"  date {i}/{len(all_dates)} ({pd.Timestamp(t).date()})")

            batch_tickers: list[str] = []
            batch_seqs: list[np.ndarray] = []
            for ticker, (dates_arr, feats_arr) in ticker_arrays.items():
                # Anchor index: last position with date <= t
                idx = int(np.searchsorted(dates_arr, t_ns, side="right")) - 1
                if idx < seq_len - 1:
                    continue
                start = idx - seq_len + 1
                seq = feats_arr[start: idx + 1]
                if seq.shape[0] != seq_len:
                    continue
                batch_tickers.append(ticker)
                batch_seqs.append(seq)

            if not batch_seqs:
                continue

            x = torch.from_numpy(np.stack(batch_seqs)).to(dm.device)
            output = model(x)
            preds = output["prediction"].squeeze(-1).cpu().numpy()
            conf_tensor = output.get("confidence")
            confs = (
                conf_tensor.squeeze(-1).cpu().numpy()
                if conf_tensor is not None
                else np.full_like(preds, 0.5)
            )
            t_stamp = pd.Timestamp(t)
            for tk, pr, cf in zip(batch_tickers, preds, confs):
                predictions.append({
                    "date": t_stamp, "ticker": tk,
                    "vol_score": float(pr), "confidence": float(cf),
                })

    return pd.DataFrame(predictions)


def main():
    setup_logger()
    cfg = load_config()
    output_dir = Path("data/research")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    ohlcv = pd.read_parquet("v3/data/raw/ohlcv_raw.parquet")
    macro = pd.read_parquet("v3/data/raw/macro.parquet")
    logger.info(
        f"OHLCV: {len(ohlcv)} rows, {ohlcv['ticker'].nunique()} tickers, "
        f"{ohlcv['date'].min()} ~ {ohlcv['date'].max()}"
    )
    logger.info(f"Macro: {macro.shape}")

    # 1. OHLCV panel — passthrough copy
    ohlcv_path = output_dir / "ohlcv_panel.parquet"
    ohlcv.to_parquet(ohlcv_path, index=False)
    logger.info(f"[1/3] OHLCV panel → {ohlcv_path}")

    # 2. Macro percentiles
    macro_pctl = build_macro_pctl(macro, ohlcv)
    macro_path = output_dir / "macro_pctl.parquet"
    macro_pctl.to_parquet(macro_path)
    logger.info(f"[2/3] Macro pctl {macro_pctl.shape} → {macro_path}")

    # 3. Vol predictions — feature compute + per-date batched inference
    logger.info("Computing features...")
    fe = VolFeatureEngineer()
    df_feats = fe.compute_all(ohlcv)
    logger.info(f"Features: {df_feats.shape}")

    storage = StorageManager(base_dir="v3")
    feat_cfg = storage.load_json("feature_config.json")
    feature_cols = feat_cfg["feature_cols"]
    logger.info(f"Feature cols: {len(feature_cols)}")

    pred_df = build_vol_predictions(df_feats, feature_cols, cfg, storage)
    pred_path = output_dir / "vol_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info(
        f"[3/3] Vol predictions {len(pred_df)} rows → {pred_path}"
    )

    print("\n=== prep_calibration_inputs complete ===")
    print(f"  ohlcv:      {ohlcv_path}  ({len(ohlcv):,} rows)")
    print(f"  macro_pctl: {macro_path}  {macro_pctl.shape}")
    print(f"  vol_pred:   {pred_path}  ({len(pred_df):,} rows)")
    print("\nNext: run_calibration_pipeline.py")


if __name__ == "__main__":
    main()
