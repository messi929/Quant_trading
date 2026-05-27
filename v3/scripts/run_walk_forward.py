"""CLI: B1 Walk-forward backtest with per-fold model retraining.

V4_ROADMAP B1: Sharpe 4.03 overfitting 판정용. 252d train / 63d test rolling.

Usage:
    PYTHONPATH=. python v3/scripts/run_walk_forward.py --epochs 5 --max-folds 2
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from v3.config.schema import load_config
from v3.backtest.walk_forward import WalkForwardValidator
from v3.utils.storage import StorageManager
from v3.utils.logger import setup_logger
from v3.utils.device import set_seed


def main():
    p = argparse.ArgumentParser(description="V3 Walk-forward validation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--test-days", type=int, default=63)
    p.add_argument("--step-days", type=int, default=63)
    p.add_argument("--epochs", type=int, default=10, help="Epochs per fold (lower = faster)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=0, help="Cap folds (0 = all)")
    p.add_argument("--output", type=str, default="v3/saved_models/walk_forward_result.json")
    args = p.parse_args()

    setup_logger()
    set_seed(42)

    cfg = load_config(args.config)
    storage = StorageManager(base_dir="v3")

    logger.info("Loading processed data...")
    df = storage.load_parquet("processed_data")
    feat_cfg = storage.load_json("feature_config.json")
    feature_cols = feat_cfg["feature_cols"]

    logger.info(f"Walk-forward config: train={args.train_days}d, test={args.test_days}d, "
                f"step={args.step_days}d, epochs={args.epochs}, max_folds={args.max_folds or 'all'}")

    validator = WalkForwardValidator(cfg)

    result = validator.run(
        df, feature_cols,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        train_epochs=args.epochs,
        patience=args.patience,
        max_folds=args.max_folds,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info(f"Result saved: {output_path}")
    logger.info(f"Summary: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
