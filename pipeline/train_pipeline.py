"""Full training pipeline orchestrating all model phases."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from data.collector import MarketDataCollector
from data.processor import DataProcessor
from data.sector_classifier import SectorClassifier
from data.feature_engineer import FeatureEngineer
from data.dataset import create_dataloaders

from models.autoencoder.model import MarketVAE
from models.autoencoder.trainer import VAETrainer
from models.transformer.model import TemporalTransformer
from models.transformer.trainer import TransformerTrainer
from models.gan.model import ConditionalWGAN
from models.gan.trainer import GANTrainer
from models.rl.environment import SectorTradingEnv
from models.rl.agent import PPOAgent
from models.rl.trainer import RLTrainer
from models.ensemble import ModelEnsemble

from indicators.generator import IndicatorGenerator
from indicators.evaluator import IndicatorEvaluator
from indicators.registry import IndicatorRegistry

from utils.device import DeviceManager, set_seed
from utils.logger import setup_logger


class TrainPipeline:
    """Orchestrates the complete training pipeline.

    Phase 1: Data collection and processing
    Phase 2: VAE training → latent indicator discovery
    Phase 3: Transformer training → temporal pattern learning
    Phase 4: GAN training → market simulation
    Phase 5: RL training → sector allocation
    Phase 6: Ensemble integration
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Resolve model_config path from settings or default
        model_config_path = self.config.get("model_config_path", "config/model_config.yaml")
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.model_config = yaml.safe_load(f)
        logger.info(f"Using model config: {model_config_path}")

        setup_logger(
            log_dir=self.config["paths"]["logs"],
            level=self.config["logging"]["level"],
        )

        set_seed(self.config["training"]["seed"])

        self.dm = DeviceManager(
            memory_fraction=self.config["gpu"]["memory_fraction"],
            compile_model=self.config["gpu"]["compile_model"],
        )

        self.sector_classifier = SectorClassifier()

    def run(self, skip_collection: bool = False, start_phase: int = 1) -> dict:
        """Run the complete training pipeline.

        Args:
            skip_collection: Skip data collection if data already exists
            start_phase: Phase to start from (1-6). Phases before this
                         will load from saved checkpoints/data.

        Returns:
            Dict with training results and metrics
        """
        results = {}

        # Phase 1: Data
        if start_phase <= 1:
            logger.info("=" * 60)
            logger.info("PHASE 1: Data Pipeline")
            logger.info("=" * 60)
            df, feature_cols, sector_map = self._phase1_data(skip_collection)
        else:
            logger.info("=" * 60)
            logger.info("PHASE 1: Loading cached data")
            logger.info("=" * 60)
            df, feature_cols, sector_map = self._load_cached_data()

        results["data"] = {"n_rows": len(df), "n_features": len(feature_cols)}

        # Create dataloaders
        seq_len = self.config["data"]["sequence_length"]
        batch_size = self.config["training"]["physical_batch_size"]

        train_loader, val_loader, test_loader = create_dataloaders(
            df, feature_cols, sector_map,
            seq_length=seq_len,
            prediction_horizon=self.model_config["transformer"]["prediction_horizon"],
            batch_size=batch_size,
            train_ratio=self.config["data"]["train_ratio"],
            val_ratio=self.config["data"]["val_ratio"],
            num_workers=self.config["training"]["num_workers"],
            pin_memory=self.config["training"]["pin_memory"],
        )

        n_features = len(feature_cols)

        # Phase 2: VAE
        if start_phase <= 2:
            logger.info("=" * 60)
            logger.info("PHASE 2: VAE - Latent Indicator Discovery")
            logger.info("=" * 60)
            vae, vae_results = self._phase2_vae(
                train_loader, val_loader, n_features
            )
            results["vae"] = vae_results
        else:
            logger.info("=" * 60)
            logger.info("PHASE 2: Loading saved VAE checkpoint")
            logger.info("=" * 60)
            vae = self._load_vae(n_features)
            results["vae"] = {"loaded_from_checkpoint": True}

        # Phase 3: Transformer
        if start_phase <= 3:
            logger.info("=" * 60)
            logger.info("PHASE 3: Transformer - Temporal Patterns")
            logger.info("=" * 60)
            transformer, tf_results = self._phase3_transformer(
                train_loader, val_loader, n_features
            )
            results["transformer"] = tf_results
        else:
            logger.info("=" * 60)
            logger.info("PHASE 3: Loading saved Transformer checkpoint")
            logger.info("=" * 60)
            transformer = self._load_transformer(n_features)
            results["transformer"] = {"loaded_from_checkpoint": True}

        # Phase 4: GAN
        if start_phase <= 4:
            logger.info("=" * 60)
            logger.info("PHASE 4: GAN - Market Simulation")
            logger.info("=" * 60)
            gan, gan_results = self._phase4_gan(
                train_loader, val_loader, n_features
            )
            results["gan"] = gan_results
        else:
            logger.info("=" * 60)
            logger.info("PHASE 4: Loading saved GAN checkpoint")
            logger.info("=" * 60)
            gan = self._load_gan(n_features)
            results["gan"] = {"loaded_from_checkpoint": True}

        # Phase 5: RL
        if start_phase <= 5:
            logger.info("=" * 60)
            logger.info("PHASE 5: RL Agent - Sector Trading")
            logger.info("=" * 60)
            rl_agent, rl_results = self._phase5_rl(
                df, feature_cols, sector_map, vae, transformer
            )
            results["rl"] = rl_results
        else:
            logger.info("=" * 60)
            logger.info("PHASE 5: Loading saved RL agent")
            logger.info("=" * 60)
            rl_agent = self._load_rl_agent(df, feature_cols, sector_map)
            results["rl"] = {"loaded_from_checkpoint": True}

        # Phase 6: Ensemble
        logger.info("=" * 60)
        logger.info("PHASE 6: Ensemble Integration")
        logger.info("=" * 60)
        ensemble = ModelEnsemble(
            vae, transformer, gan, rl_agent,
            device=self.dm.device,
        )
        ensemble.save_ensemble(
            str(Path(self.config["paths"]["models"]) / "ensemble.pt")
        )
        results["ensemble"] = {"saved": True}

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results

    def _load_cached_data(self) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """Load processed data from cache."""
        processed_path = Path(self.config["paths"]["processed_data"]) / "processed_data.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Cached data not found at {processed_path}. Run from phase 1 first."
            )

        df = pd.read_parquet(processed_path)

        # Infer feature columns by excluding metadata columns
        meta_cols = {"date", "open", "high", "low", "close", "volume", "ticker", "market", "sector"}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        sector_map = {
            name: idx
            for idx, name in enumerate(self.sector_classifier.sector_names)
        }

        logger.info(
            f"Loaded cached data: {len(df)} rows, "
            f"{df['ticker'].nunique()} tickers, "
            f"{len(feature_cols)} features"
        )
        return df, feature_cols, sector_map

    def _load_vae(self, n_features: int) -> MarketVAE:
        """Load VAE model from latest checkpoint."""
        from utils.storage import StorageManager

        cfg = self.model_config["vae"]
        seq_len = self.config["data"]["sequence_length"]

        model = MarketVAE(
            input_dim=n_features,
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
            seq_length=seq_len,
        )

        storage = StorageManager()
        checkpoint = storage.load_model_checkpoint("vae")
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.dm.device)
        model.eval()
        logger.info(f"VAE loaded (epoch {checkpoint['epoch']})")
        return model

    def _load_transformer(self, n_features: int) -> TemporalTransformer:
        """Load Transformer model from latest checkpoint."""
        from utils.storage import StorageManager

        cfg = self.model_config["transformer"]

        model = TemporalTransformer(
            input_dim=n_features,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_encoder_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_length=cfg["max_seq_length"],
            output_dim=cfg["output_dim"],
            n_sectors=cfg["n_sectors"],
            use_sector_attention=cfg["use_sector_attention"],
            sector_embed_dim=cfg["sector_embed_dim"],
        )

        storage = StorageManager()
        checkpoint = storage.load_model_checkpoint("transformer")
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.dm.device)
        model.eval()
        logger.info(f"Transformer loaded (epoch {checkpoint['epoch']})")
        return model

    def _load_gan(self, n_features: int) -> ConditionalWGAN:
        """Load GAN model from latest checkpoint."""
        from utils.storage import StorageManager

        cfg = self.model_config["gan"]

        model = ConditionalWGAN(
            noise_dim=cfg["noise_dim"],
            hidden_dims=cfg["generator_hidden_dims"],
            output_dim=n_features,
            seq_length=cfg["sequence_length"],
        )

        storage = StorageManager()
        checkpoint = storage.load_model_checkpoint("gan")
        sd = checkpoint["state_dict"]
        # Checkpoint saved generator/discriminator separately (see gan/trainer.py)
        model.generator.load_state_dict(sd["generator"])
        model.discriminator.load_state_dict(sd["discriminator"])
        model.to(self.dm.device)
        model.eval()
        logger.info(f"GAN loaded (epoch {checkpoint['epoch']})")
        return model

    def _load_rl_agent(
        self, df: pd.DataFrame, feature_cols: list[str], sector_map: dict[str, int],
    ) -> PPOAgent:
        """Load RL agent from latest checkpoint."""
        cfg = self.model_config["rl"]
        # Compute state_dim the same way as _phase5_rl
        state_dim = len(feature_cols) + len(sector_map) + 2
        agent = PPOAgent(
            state_dim=state_dim,
            n_sectors=len(sector_map),
            hidden_dims=cfg["hidden_dims"],
            device=self.dm.device,
        )
        # Try loading checkpoint if available
        try:
            from utils.storage import StorageManager
            storage = StorageManager()
            checkpoint = storage.load_model_checkpoint("rl")
            agent.actor.load_state_dict(checkpoint["state_dict"]["actor"])
            agent.critic.load_state_dict(checkpoint["state_dict"]["critic"])
            logger.info(f"RL agent loaded (epoch {checkpoint['epoch']})")
        except FileNotFoundError:
            logger.warning("No RL checkpoint found, using untrained agent")
        return agent

    def _phase1_data(
        self, skip_collection: bool,
    ) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """Data collection, processing, and feature engineering."""
        raw_dir = Path(self.config["paths"]["raw_data"])

        if not skip_collection or not (raw_dir / "kospi_ohlcv.parquet").exists():
            collector = MarketDataCollector(
                save_dir=str(raw_dir),
                history_years=self.config["data"]["history_years"],
            )
            market_data = collector.collect_all()
            raw_df = pd.concat(market_data.values(), ignore_index=True)
        else:
            logger.info("Loading existing raw data...")
            dfs = []
            for f in raw_dir.glob("*_ohlcv.parquet"):
                dfs.append(pd.read_parquet(f))
            raw_df = pd.concat(dfs, ignore_index=True)

        # Process
        processor = DataProcessor(
            min_history_days=self.config["data"]["min_history_days"]
        )
        df = processor.process(raw_df)

        # Sector classification
        ticker_info = df[["ticker", "market"]].drop_duplicates()
        # Load actual company names for KOSPI keyword matching
        ticker_info_path = raw_dir / "ticker_info.parquet"
        if ticker_info_path.exists():
            name_df = pd.read_parquet(ticker_info_path)[["ticker", "name"]]
            ticker_info = ticker_info.merge(name_df, on="ticker", how="left")
            ticker_info["name"] = ticker_info["name"].fillna(ticker_info["ticker"])
        else:
            ticker_info["name"] = ticker_info["ticker"]

        kospi_tickers = ticker_info[ticker_info["market"] == "KOSPI"]
        nasdaq_tickers = ticker_info[ticker_info["market"] == "NASDAQ"]

        if len(kospi_tickers) > 0:
            kospi_classified = self.sector_classifier.classify_kospi(kospi_tickers)
        else:
            kospi_classified = pd.DataFrame()

        if len(nasdaq_tickers) > 0:
            nasdaq_classified = self.sector_classifier.classify_nasdaq(nasdaq_tickers)
        else:
            nasdaq_classified = pd.DataFrame()

        classified = pd.concat(
            [kospi_classified, nasdaq_classified], ignore_index=True
        )
        sector_mapping = dict(zip(classified["ticker"], classified["sector"]))
        df["sector"] = df["ticker"].map(sector_mapping).fillna("unknown")

        sector_map = {
            name: idx
            for idx, name in enumerate(self.sector_classifier.sector_names)
        }

        # Filter out unknown sector tickers if configured
        if self.config["data"].get("filter_unknown_sectors", False):
            before = df["ticker"].nunique()
            df = df[df["sector"] != "unknown"]
            after = df["ticker"].nunique()
            logger.info(f"Filtered unknown sectors: {before} -> {after} tickers")

        # Limit tickers per market for fast training if configured
        max_tickers = self.config["training"].get("max_tickers")
        if max_tickers and df["ticker"].nunique() > max_tickers:
            # Keep tickers with most data points (proxy for liquidity)
            ticker_counts = df.groupby("ticker").size().nlargest(max_tickers)
            df = df[df["ticker"].isin(ticker_counts.index)]
            logger.info(f"Limited to top {max_tickers} tickers by data coverage")

        # Feature engineering
        feature_eng = FeatureEngineer()
        df = feature_eng.compute_all(df)
        feature_cols = feature_eng.get_feature_names()

        # Normalize
        df = processor.normalize(df, feature_cols, fit=True)

        # Save processed data
        processed_dir = Path(self.config["paths"]["processed_data"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_dir / "processed_data.parquet")

        logger.info(
            f"Phase 1 complete: {len(df)} rows, "
            f"{df['ticker'].nunique()} tickers, "
            f"{len(feature_cols)} features"
        )
        return df, feature_cols, sector_map

    def _phase2_vae(
        self,
        train_loader,
        val_loader,
        n_features: int,
    ) -> tuple:
        """Train VAE for latent indicator discovery."""
        cfg = self.model_config["vae"]
        seq_len = self.config["data"]["sequence_length"]

        model = MarketVAE(
            input_dim=n_features,
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
            seq_length=seq_len,
        )

        trainer = VAETrainer(
            model, self.dm,
            learning_rate=cfg["learning_rate"],
            kl_weight=cfg["kl_weight"],
            kl_annealing=cfg["kl_annealing"],
            kl_annealing_epochs=cfg["kl_annealing_epochs"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
        )

        results = trainer.train(
            train_loader, val_loader,
            epochs=cfg["epochs"],
            patience=self.config["training"]["early_stopping_patience"],
        )

        # Extract latent features and evaluate indicators
        logger.info("Extracting latent indicators...")
        latent_features, sector_ids = trainer.extract_latent_features(val_loader)

        self.dm.clear_memory()
        return model, results

    def _phase3_transformer(
        self,
        train_loader,
        val_loader,
        n_features: int,
    ) -> tuple:
        """Train Temporal Transformer."""
        cfg = self.model_config["transformer"]

        model = TemporalTransformer(
            input_dim=n_features,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_encoder_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_length=cfg["max_seq_length"],
            output_dim=cfg["output_dim"],
            n_sectors=cfg["n_sectors"],
            use_sector_attention=cfg["use_sector_attention"],
            sector_embed_dim=cfg["sector_embed_dim"],
        )

        trainer = TransformerTrainer(
            model, self.dm,
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg["warmup_steps"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
        )

        results = trainer.train(
            train_loader, val_loader,
            epochs=cfg["epochs"],
            patience=self.config["training"]["early_stopping_patience"],
        )

        self.dm.clear_memory()
        return model, results

    def _phase4_gan(
        self,
        train_loader,
        val_loader,
        n_features: int,
    ) -> tuple:
        """Train Conditional WGAN-GP."""
        cfg = self.model_config["gan"]

        model = ConditionalWGAN(
            noise_dim=cfg["noise_dim"],
            hidden_dims=cfg["generator_hidden_dims"],
            output_dim=n_features,
            seq_length=cfg["sequence_length"],
        )

        trainer = GANTrainer(
            model, self.dm,
            lr_g=cfg["learning_rate_g"],
            lr_d=cfg["learning_rate_d"],
            n_critic=cfg["n_critic"],
            gp_weight=cfg["gradient_penalty_weight"],
        )

        results = trainer.train(
            train_loader, val_loader,
            epochs=cfg["epochs"],
        )

        self.dm.clear_memory()
        return model, results

    def _phase5_rl(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        sector_map: dict[str, int],
        vae: MarketVAE,
        transformer: TemporalTransformer,
    ) -> tuple:
        """Train RL agent for sector trading."""
        cfg = self.model_config["rl"]

        # Prepare sector returns for RL environment
        # IMPORTANT: return_1d is RobustScaler-normalized, NOT actual returns.
        # Use actual close price pct_change to get real decimal returns.
        sector_names = list(sector_map.keys())
        sector_returns = []
        for sector in sector_names:
            sector_data = df[df["sector"] == sector].sort_values("date")
            if "close" in sector_data.columns:
                close_pivot = sector_data.pivot_table(
                    index="date", columns="ticker", values="close"
                )
                stock_rets = close_pivot.pct_change()
                avg_ret = stock_rets.mean(axis=1)
                avg_ret.name = sector
                sector_returns.append(avg_ret)

        if not sector_returns:
            logger.warning("No sector returns available for RL training")
            state_dim = len(feature_cols) + len(sector_map) + 2
            agent = PPOAgent(state_dim, len(sector_map), device=self.dm.device)
            return agent, {"skipped": True}

        sector_ret_df = pd.concat(sector_returns, axis=1)
        sector_ret_df.columns = sector_names[:len(sector_returns)]
        sector_ret_df = sector_ret_df.fillna(0)

        # Prepare features for environment (use mean features per day)
        daily_features = df.groupby("date")[feature_cols].mean().values
        n_env_steps = min(len(daily_features), len(sector_ret_df))
        daily_features = daily_features[:n_env_steps]
        sector_ret_array = sector_ret_df.values[:n_env_steps]

        # Pad sector returns if fewer sectors than expected
        n_actual_sectors = sector_ret_array.shape[1]
        if n_actual_sectors < len(sector_map):
            pad = np.zeros((n_env_steps, len(sector_map) - n_actual_sectors))
            sector_ret_array = np.concatenate([sector_ret_array, pad], axis=1)

        env = SectorTradingEnv(
            features=daily_features,
            sector_returns=sector_ret_array,
            n_sectors=len(sector_map),
            initial_capital=self.config["backtest"]["initial_capital"],
            transaction_cost=cfg["transaction_cost"],
            max_position=cfg["max_position"],
            reward_type=cfg["reward_type"],
            risk_penalty=cfg["risk_penalty"],
            drawdown_penalty=cfg["drawdown_penalty"],
            window_size=cfg.get("window_size", 60),
            vol_target=cfg.get("vol_target", 0.15),
            episode_length=cfg.get("episode_length", 252),
        )

        state_dim = env.observation_space.shape[0]
        agent = PPOAgent(
            state_dim=state_dim,
            n_sectors=len(sector_map),
            hidden_dims=cfg["hidden_dims"],
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_epsilon=cfg["clip_epsilon"],
            value_coef=cfg["value_coef"],
            entropy_coef=cfg["entropy_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            device=self.dm.device,
        )

        rl_trainer = RLTrainer(
            agent, env,
            n_steps=cfg["n_steps"],
            total_timesteps=cfg["total_timesteps"],
        )

        results = rl_trainer.train()
        eval_results = rl_trainer.evaluate(n_episodes=10)
        results["evaluation"] = eval_results

        self.dm.clear_memory()
        return agent, results
