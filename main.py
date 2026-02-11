"""
Alpha Signal Discovery Engine - Main Entry Point

Discovers new mathematical indicators from KOSPI/NASDAQ market data
using deep learning (VAE + Transformer + GAN + RL), then trades
sector-based portfolios using these discovered signals.

Usage:
    python main.py train              # Full training pipeline
    python main.py train --skip-data  # Skip data collection
    python main.py infer              # Run inference
    python main.py backtest           # Run backtest on test data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from loguru import logger

from utils.logger import setup_logger
from utils.device import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alpha Signal Discovery Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data collection (use existing data)",
    )
    train_parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings config",
    )

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--ensemble", default="saved_models/ensemble.pt",
        help="Path to ensemble checkpoint",
    )
    infer_parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings config",
    )

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings config",
    )
    bt_parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward optimization",
    )

    return parser.parse_args()


def cmd_train(args):
    """Execute training pipeline."""
    from pipeline.train_pipeline import TrainPipeline

    pipeline = TrainPipeline(config_path=args.config)
    results = pipeline.run(skip_collection=args.skip_data)

    logger.info("Training Results Summary:")
    for phase, metrics in results.items():
        logger.info(f"  {phase}: {metrics}")


def cmd_infer(args):
    """Execute inference pipeline."""
    from pipeline.inference_pipeline import InferencePipeline

    pipeline = InferencePipeline(
        config_path=args.config,
        ensemble_path=args.ensemble,
    )

    signals = pipeline.generate_signals()

    logger.info("Current Sector Allocations:")
    for sector, weight in signals.get("sector_allocations", {}).items():
        direction = "LONG" if weight > 0 else "SHORT" if weight < 0 else "FLAT"
        logger.info(f"  {sector}: {weight:+.4f} ({direction})")

    logger.info(f"Confidence: {signals.get('confidence', 0):.2%}")
    logger.info(f"Risk Level: {signals.get('risk_report', {}).get('risk_level', 'N/A')}")


def cmd_backtest(args):
    """Execute backtesting."""
    from backtest.engine import BacktestEngine
    from backtest.visualizer import BacktestVisualizer

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_path = Path(config["paths"]["processed_data"]) / "processed_data.parquet"
    if not processed_path.exists():
        logger.error(
            f"Processed data not found at {processed_path}. "
            f"Run 'python main.py train' first."
        )
        return

    import pandas as pd
    df = pd.read_parquet(processed_path)

    # Build sector returns
    sector_returns_list = []
    sectors = df["sector"].unique()
    for sector in sectors:
        sector_data = df[df["sector"] == sector]
        if "return_1d" in sector_data.columns:
            avg_ret = sector_data.groupby("date")["return_1d"].mean()
            avg_ret.name = sector
            sector_returns_list.append(avg_ret)

    if not sector_returns_list:
        logger.error("No sector returns available for backtesting")
        return

    sector_returns = pd.concat(sector_returns_list, axis=1).fillna(0)
    sector_returns.index = pd.to_datetime(sector_returns.index)

    # Use test period only
    n_dates = len(sector_returns)
    test_start = int(n_dates * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    test_returns = sector_returns.iloc[test_start:]

    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital"],
        commission_rate=config["backtest"]["commission_rate"],
        slippage_rate=config["backtest"]["slippage_rate"],
        rebalance_frequency=config["backtest"]["rebalance_frequency"],
    )

    # Simple equal-weight signals for baseline
    n_sectors = test_returns.shape[1]
    equal_signals = np.ones((len(test_returns), n_sectors)) / n_sectors

    if args.walk_forward:
        results = engine.walk_forward(test_returns, equal_signals)
        logger.info(f"Walk-forward: {len(results)} periods")
        for i, r in enumerate(results):
            logger.info(
                f"  Period {i+1}: Return={r['metrics']['total_return']:.2%}, "
                f"Sharpe={r['metrics']['sharpe_ratio']:.2f}"
            )
    else:
        result = engine.run(test_returns, equal_signals)
        viz = BacktestVisualizer(save_dir=config["paths"].get("results", "results"))
        viz.generate_report(result, sector_names=list(test_returns.columns))

        logger.info("Backtest Results:")
        for key, val in result["metrics"].items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")


def main():
    args = parse_args()

    if args.command is None:
        logger.info("No command specified. Use --help for options.")
        logger.info("  python main.py train       # Full training pipeline")
        logger.info("  python main.py infer       # Run inference")
        logger.info("  python main.py backtest    # Run backtest")
        return

    with open(getattr(args, "config", "config/settings.yaml"), "r") as f:
        config = yaml.safe_load(f)

    setup_logger(
        log_dir=config["paths"]["logs"],
        level=config["logging"]["level"],
    )
    set_seed(config["training"]["seed"])

    commands = {
        "train": cmd_train,
        "infer": cmd_infer,
        "backtest": cmd_backtest,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        logger.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
