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

import numpy as np
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
        "--start-phase", type=int, default=1, choices=range(1, 7),
        help="Phase to start from (1-6). Earlier phases load from checkpoints.",
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
    results = pipeline.run(
        skip_collection=args.skip_data or args.start_phase > 1,
        start_phase=args.start_phase,
    )

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
    """Execute backtesting with trained ensemble model signals."""
    import pandas as pd
    import torch
    from backtest.engine import BacktestEngine
    from backtest.visualizer import BacktestVisualizer

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve model_config path relative to settings config
    config_dir = Path(args.config).parent
    model_config_path = config.get("model_config_path", str(config_dir / "model_config.yaml"))
    # If settings_fast.yaml is used, try model_config_fast.yaml
    if "fast" in Path(args.config).stem and not Path(model_config_path).exists():
        model_config_path = str(config_dir / "model_config_fast.yaml")
    if not Path(model_config_path).exists():
        model_config_path = "config/model_config.yaml"

    # Load processed data
    processed_path = Path(config["paths"]["processed_data"]) / "processed_data.parquet"
    if not processed_path.exists():
        logger.error(
            f"Processed data not found at {processed_path}. "
            f"Run 'python main.py train' first."
        )
        return

    df = pd.read_parquet(processed_path)

    # Build sector returns from ACTUAL close prices (not normalized return_1d)
    # return_1d in processed data is RobustScaler-normalized, so we must
    # compute actual decimal returns from close prices instead.
    sector_returns_list = []
    sectors = sorted(df["sector"].unique())
    for sector in sectors:
        sector_data = df[df["sector"] == sector]
        # Pivot to (date x ticker) close price table, then compute pct_change per ticker
        close_pivot = sector_data.pivot_table(index="date", columns="ticker", values="close")
        stock_returns = close_pivot.pct_change()
        # Equal-weight average actual return for the sector each day
        avg_ret = stock_returns.mean(axis=1)
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

    # --- Try to load trained ensemble for model-based signals ---
    ensemble_path = Path(config["paths"]["models"]) / "ensemble.pt"
    model_signals = None

    if ensemble_path.exists():
        logger.info(f"Loading trained ensemble from {ensemble_path}")
        try:
            from models.autoencoder.model import MarketVAE
            from models.transformer.model import TemporalTransformer
            from models.gan.model import ConditionalWGAN
            from models.rl.agent import PPOAgent
            from models.ensemble import ModelEnsemble
            from utils.device import DeviceManager

            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = yaml.safe_load(f)

            dm = DeviceManager(
                memory_fraction=config["gpu"]["memory_fraction"],
                compile_model=False,
            )

            # Derive feature columns from the already-loaded processed data,
            # matching the same exclusion logic used in FeatureEngineer.compute_all()
            _meta_cols = {"date", "open", "high", "low", "close", "volume", "ticker", "market", "sector"}
            feature_cols = [c for c in df.columns if c not in _meta_cols]
            n_features = len(feature_cols)
            seq_len = config["data"]["sequence_length"]

            vae_cfg = model_config["vae"]
            tf_cfg = model_config["transformer"]
            gan_cfg = model_config["gan"]
            rl_cfg = model_config["rl"]

            vae = MarketVAE(
                input_dim=n_features,
                hidden_dims=vae_cfg["hidden_dims"],
                latent_dim=vae_cfg["latent_dim"],
                dropout=vae_cfg.get("dropout", 0.2),
                seq_length=seq_len,
            )
            transformer = TemporalTransformer(
                input_dim=n_features,
                d_model=tf_cfg["d_model"],
                n_heads=tf_cfg["n_heads"],
                n_layers=tf_cfg["n_encoder_layers"],
                d_ff=tf_cfg["d_ff"],
                dropout=tf_cfg.get("dropout", 0.1),
                max_seq_length=tf_cfg["max_seq_length"],
                output_dim=tf_cfg.get("output_dim", 1),
                n_sectors=tf_cfg["n_sectors"],
                use_sector_attention=tf_cfg.get("use_sector_attention", True),
                sector_embed_dim=tf_cfg.get("sector_embed_dim", 32),
            )
            gan = ConditionalWGAN(
                noise_dim=gan_cfg["noise_dim"],
                hidden_dims=gan_cfg.get("generator_hidden_dims", [128, 256, 128]),
                output_dim=n_features,
                seq_length=gan_cfg["sequence_length"],
            )

            # state_dim must match training: n_features + n_sectors + 2 (portfolio state)
            n_sectors_rl = len(sorted(df["sector"].unique()))
            state_dim = n_features + n_sectors_rl + 2
            agent = PPOAgent(
                state_dim=state_dim,
                n_sectors=n_sectors_rl,
                hidden_dims=rl_cfg["hidden_dims"],
                device=dm.device,
            )

            ensemble = ModelEnsemble(vae, transformer, gan, agent, device=dm.device)
            ensemble.load_ensemble(str(ensemble_path))
            logger.info("Ensemble loaded — generating model-based signals for backtest")

            # Generate signals for each test day using rolling windows
            n_test = len(test_returns)
            n_sectors = test_returns.shape[1]
            model_signals = np.zeros((n_test, n_sectors))

            # Build sector→id map matching training order (sectors.yaml key order)
            _sectors_cfg_path = config_dir / "sectors.yaml"
            with open(_sectors_cfg_path, "r", encoding="utf-8") as _sf:
                _sectors_cfg = yaml.safe_load(_sf)
            _train_sector_names = list(_sectors_cfg["sectors"].keys())
            _sector_to_train_id = {name: idx for idx, name in enumerate(_train_sector_names)}

            # Use training-time sector ID (sectors.yaml order), not sorted index
            # Per-ticker inference: run model on each ticker individually, then
            # average signals to the sector level (matches training distribution)
            for s_idx, sector in enumerate(test_returns.columns):
                train_sector_id = _sector_to_train_id.get(sector, 0)
                sector_id_tensor = torch.tensor([train_sector_id], dtype=torch.long).to(dm.device)

                sector_data = df[df["sector"] == sector].sort_values("date")
                tickers_in_sector = sector_data["ticker"].unique()

                # Accumulate predictions per ticker for each test day
                ticker_signal_sum = np.zeros((n_test,))
                ticker_signal_count = np.zeros((n_test,), dtype=int)

                for ticker in tickers_in_sector:
                    tkr_data = sector_data[sector_data["ticker"] == ticker].sort_values("date")
                    if len(tkr_data) < seq_len:
                        continue
                    tkr_features = tkr_data[feature_cols].values
                    tkr_dates = pd.to_datetime(tkr_data["date"].values)

                    for t_idx, test_date in enumerate(test_returns.index):
                        mask = tkr_dates <= test_date
                        valid_count = mask.sum()
                        if valid_count < seq_len:
                            continue
                        window = tkr_features[valid_count - seq_len : valid_count]
                        seq_tensor = torch.FloatTensor(window).unsqueeze(0).to(dm.device)
                        with torch.no_grad():
                            signals_out = ensemble.get_signals(seq_tensor, sector_id_tensor)
                            ticker_signal_sum[t_idx] += signals_out["prediction"].cpu().item()
                            ticker_signal_count[t_idx] += 1

                # Average over tickers; leave 0 where no ticker has enough data
                valid = ticker_signal_count > 0
                model_signals[valid, s_idx] = ticker_signal_sum[valid] / ticker_signal_count[valid]

            # Normalize signals: top-K sectors long, rest neutral
            # Avoids always-100%-long problem; negative signals treated as neutral
            top_k = 5
            for t_idx in range(n_test):
                row = model_signals[t_idx]
                weights = np.zeros_like(row)
                if np.any(row != 0):
                    # Select top-K sectors by signal strength
                    top_indices = np.argsort(row)[-top_k:]
                    weights[top_indices] = row[top_indices]
                    # Only keep positive signals (negative = no conviction)
                    weights = np.clip(weights, 0, None)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        # All top-K signals were negative → equal weight
                        weights = np.ones(n_sectors) / n_sectors
                else:
                    weights = np.ones(n_sectors) / n_sectors
                # Blend model signal with equal-weight: model adds tilts on top of
                # equal-weight base, reducing concentration risk
                alpha = 0.4  # 40% model signal, 60% equal-weight
                ew = np.ones(n_sectors) / n_sectors
                weights = alpha * weights + (1.0 - alpha) * ew
                model_signals[t_idx] = weights

            # Apply daily risk management pre-filter to signals
            # This complements the weekly-rebalancing engine's own RiskManager
            # by providing daily circuit-breaker protection
            from strategy.risk import RiskManager as _RM
            pre_risk = _RM(max_position_pct=1.0)  # No per-sector clipping here
            sim_portfolio = float(config["backtest"]["initial_capital"])
            for t_idx in range(n_test):
                weights = model_signals[t_idx].copy()
                recent_ret = (
                    test_returns.iloc[max(0, t_idx - 20) : t_idx + 1]
                    if t_idx > 0 else None
                )
                adj_weights, _ = pre_risk.check_and_adjust(
                    weights, sim_portfolio, recent_ret
                )
                model_signals[t_idx] = adj_weights
                # Simulate one-day return to track portfolio value
                day_ret = float(np.dot(adj_weights, test_returns.iloc[t_idx].values))
                sim_portfolio *= (1.0 + day_ret)

            dm.clear_memory()
            logger.info("Model signal generation complete")

        except Exception as e:
            logger.warning(f"Failed to load ensemble model: {e}")
            logger.warning("Falling back to equal-weight baseline signals")
            model_signals = None

    # Fallback: equal-weight signals
    if model_signals is None:
        logger.info("Using equal-weight baseline signals")
        n_sectors = test_returns.shape[1]
        model_signals = np.ones((len(test_returns), n_sectors)) / n_sectors

    if args.walk_forward:
        # Use 30-day burn-in + 30-day test windows to fit within the ~180-day test set
        n_test_days = len(test_returns)
        wf_train = min(30, n_test_days // 6)
        wf_test = min(30, n_test_days // 6)
        results = engine.walk_forward(test_returns, model_signals, train_window=wf_train, test_window=wf_test)
        logger.info(f"Walk-forward: {len(results)} periods (train_window={wf_train}, test_window={wf_test})")

        sharpes, returns, mdds = [], [], []
        for i, r in enumerate(results):
            m = r["metrics"]
            period = r.get("period", {})
            s = m.get("sharpe_ratio", 0)
            ret = m.get("total_return", 0)
            mdd = m.get("max_drawdown", 0)
            sharpes.append(s)
            returns.append(ret)
            mdds.append(mdd)
            logger.info(
                f"  Period {i+1} [{period.get('test_start', '')} ~ {period.get('test_end', '')}]: "
                f"Return={ret:.2%}, Sharpe={s:.2f}, MDD={mdd:.2%}"
            )

        if results:
            logger.info("\n--- Walk-forward Summary ---")
            logger.info(f"  Mean Sharpe : {np.mean(sharpes):.2f}  (std={np.std(sharpes):.2f})")
            logger.info(f"  Mean Return : {np.mean(returns):.2%} (std={np.std(returns):.2%})")
            logger.info(f"  Mean MDD    : {np.mean(mdds):.2%}")
            logger.info(f"  Profitable  : {sum(r > 0 for r in returns)}/{len(returns)} periods")
            positive_sharpe = sum(s > 0 for s in sharpes)
            logger.info(f"  Sharpe > 0  : {positive_sharpe}/{len(sharpes)} periods")
    else:
        result = engine.run(test_returns, model_signals)
        viz = BacktestVisualizer(save_dir=config["paths"].get("results", "results"))
        viz.generate_report(result, sector_names=list(test_returns.columns))

        logger.info("Backtest Results:")
        for key, val in result["metrics"].items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")

        # Also run equal-weight baseline for comparison
        if ensemble_path.exists():
            logger.info("\n--- Equal-Weight Baseline Comparison ---")
            n_sectors = test_returns.shape[1]
            baseline_signals = np.ones((len(test_returns), n_sectors)) / n_sectors
            baseline_result = engine.run(test_returns, baseline_signals)
            logger.info("Baseline Results:")
            for key in ["total_return", "sharpe_ratio", "max_drawdown", "sortino_ratio"]:
                val = baseline_result["metrics"].get(key)
                if val is not None and isinstance(val, float):
                    logger.info(f"  {key}: {val:.4f}")


def main():
    args = parse_args()

    if args.command is None:
        logger.info("No command specified. Use --help for options.")
        logger.info("  python main.py train       # Full training pipeline")
        logger.info("  python main.py infer       # Run inference")
        logger.info("  python main.py backtest    # Run backtest")
        return

    with open(getattr(args, "config", "config/settings.yaml"), "r", encoding="utf-8") as f:
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
