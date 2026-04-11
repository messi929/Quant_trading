"""CLI: Run V3 backtest with vol predictions."""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.config.schema import load_config
from v3.backtest.engine import BacktestEngine
from v3.backtest.metrics import print_report, trades_to_dataframe
from v3.data.normalizer import FeatureNormalizer
from v3.model.vol_transformer import VolTransformer
from v3.model.inference import VolInference
from v3.utils.device import DeviceManager, set_seed
from v3.utils.logger import setup_logger
from v3.utils.storage import StorageManager
from loguru import logger


def generate_vol_predictions(cfg, storage, df, feature_cols):
    """Generate vol predictions for all test-period data."""
    logger.info("Loading model for inference...")

    # Load normalizer
    normalizer = FeatureNormalizer.load(cfg.paths.normalizer_stats)

    # Load model
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
    model = model.to(dm.device)
    model.eval()

    # Normalize features
    df_norm = normalizer.transform(df, feature_cols)

    # Generate predictions for each (date, ticker)
    seq_len = cfg.data.sequence_length
    predictions = []

    dates = sorted(df_norm["date"].unique())
    # Use only test period
    test_start = int(len(dates) * (cfg.data.train_ratio + cfg.data.val_ratio))
    test_dates = dates[test_start:]

    logger.info(f"Generating predictions for {len(test_dates)} test dates...")

    with torch.no_grad():
        for date in test_dates:
            for ticker, group in df_norm.groupby("ticker"):
                group = group.sort_values("date")
                # Get data up to this date
                mask = group["date"] <= date
                available = group[mask]

                if len(available) < seq_len:
                    continue

                features = available[feature_cols].values[-seq_len:]
                features = np.nan_to_num(features, nan=0.0).astype(np.float32)

                x = torch.tensor(features).unsqueeze(0).to(dm.device)
                output = model(x)

                conf_tensor = output.get("confidence")
                conf_val = conf_tensor.item() if conf_tensor is not None else 0.5

                predictions.append({
                    "date": date,
                    "ticker": ticker,
                    "vol_score": output["prediction"].item(),
                    "confidence": conf_val,
                })

    pred_df = pd.DataFrame(predictions)
    logger.info(f"Generated {len(pred_df)} predictions")
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="V3 Backtest")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    setup_logger()
    set_seed(42)

    cfg = load_config(args.config)
    storage = StorageManager(base_dir="v3")

    # Load data
    logger.info("Loading processed data...")
    df = storage.load_parquet("processed_data")
    feat_cfg = storage.load_json("feature_config.json")
    feature_cols = feat_cfg["feature_cols"]

    # Generate predictions
    vol_predictions = generate_vol_predictions(cfg, storage, df, feature_cols)

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(cfg)
    result = engine.run(df, vol_predictions)

    # Print report
    print_report(result)

    # Save results
    if result.trades:
        trade_df = trades_to_dataframe(result.trades)
        storage.save_parquet(trade_df, "backtest_trades", subdir="results")

    if not result.equity_curve.empty:
        storage.save_parquet(result.equity_curve, "backtest_equity", subdir="results")

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
