"""Real-time inference pipeline for generating trading signals."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from data.collector import MarketDataCollector
from data.processor import DataProcessor
from data.feature_engineer import FeatureEngineer

from models.autoencoder.model import MarketVAE
from models.transformer.model import TemporalTransformer
from models.gan.model import ConditionalWGAN
from models.rl.agent import PPOAgent
from models.ensemble import ModelEnsemble

from strategy.signal import SignalGenerator
from strategy.portfolio import PortfolioOptimizer
from strategy.risk import RiskManager

from utils.device import DeviceManager


class InferencePipeline:
    """Generates live trading signals from trained models.

    Pipeline:
        1. Fetch latest market data
        2. Process and compute features
        3. Run through model ensemble
        4. Generate sector allocation signals
        5. Apply risk management
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        ensemble_path: str = "saved_models/ensemble.pt",
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_config_path = self.config.get("model_config_path", "config/model_config.yaml")
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.model_config = yaml.safe_load(f)

        self.dm = DeviceManager(
            memory_fraction=self.config["gpu"]["memory_fraction"],
            compile_model=False,  # Skip compile for inference
        )

        self.processor = DataProcessor()
        self.feature_eng = FeatureEngineer()
        self.signal_gen = SignalGenerator()
        self.portfolio_opt = PortfolioOptimizer(
            max_position_pct=self.config["backtest"]["max_position_pct"],
        )
        self.risk_mgr = RiskManager()

        # Load models
        self.ensemble = self._load_ensemble(ensemble_path)
        self.seq_length = self.config["data"]["sequence_length"]

    def _load_ensemble(self, path: str) -> ModelEnsemble:
        """Load trained model ensemble."""
        vae_cfg = self.model_config["vae"]
        tf_cfg = self.model_config["transformer"]
        gan_cfg = self.model_config["gan"]
        rl_cfg = self.model_config["rl"]

        # Placeholder input dims - will be set from saved state
        vae = MarketVAE(
            input_dim=1,  # Will be overridden by checkpoint
            latent_dim=vae_cfg["latent_dim"],
            seq_length=self.config["data"]["sequence_length"],
        )

        transformer = TemporalTransformer(
            input_dim=1,  # Will be overridden
            d_model=tf_cfg["d_model"],
            n_heads=tf_cfg["n_heads"],
            n_layers=tf_cfg["n_encoder_layers"],
            d_ff=tf_cfg["d_ff"],
            n_sectors=tf_cfg["n_sectors"],
        )

        gan = ConditionalWGAN(
            noise_dim=gan_cfg["noise_dim"],
            output_dim=1,  # Will be overridden
            seq_length=gan_cfg["sequence_length"],
        )

        state_dim = vae_cfg["latent_dim"] + tf_cfg["d_model"]  # Approximation
        agent = PPOAgent(
            state_dim=state_dim,
            n_sectors=rl_cfg["n_sectors"],
            hidden_dims=rl_cfg["hidden_dims"],
            device=self.dm.device,
        )

        ensemble = ModelEnsemble(
            vae, transformer, gan, agent,
            device=self.dm.device,
        )

        if Path(path).exists():
            ensemble.load_ensemble(path)
            logger.info(f"Loaded ensemble from {path}")
        else:
            logger.warning(f"Ensemble not found at {path}, using untrained models")

        return ensemble

    def generate_signals(
        self,
        market_data: pd.DataFrame = None,
        portfolio_value: float = None,
    ) -> dict:
        """Generate trading signals for current market state.

        Args:
            market_data: Recent market data. If None, fetches latest.
            portfolio_value: Current portfolio value for risk management.

        Returns:
            Dict with signals, allocations, and risk report
        """
        if market_data is None:
            collector = MarketDataCollector(history_years=1)
            market_data_dict = collector.collect_all(save=False)
            market_data = pd.concat(market_data_dict.values(), ignore_index=True)

        # Process
        df = self.processor.process(market_data)
        df = self.feature_eng.compute_all(df)
        feature_cols = self.feature_eng.get_feature_names()
        df = self.processor.normalize(df, feature_cols, fit=True)

        # Get latest sequences for each sector
        signals_per_sector = {}
        sector_predictions = {}

        for sector in df["sector"].unique():
            sector_data = df[df["sector"] == sector].tail(self.seq_length)
            if len(sector_data) < self.seq_length:
                continue

            features = torch.FloatTensor(
                sector_data[feature_cols].values
            ).unsqueeze(0).to(self.dm.device)

            sector_id = torch.tensor([0], dtype=torch.long).to(self.dm.device)

            with torch.no_grad():
                model_signals = self.ensemble.get_signals(features, sector_id)
                signals_per_sector[sector] = {
                    "prediction": model_signals["prediction"].cpu().item(),
                    "latent": model_signals["latent_indicators"].cpu().numpy(),
                }

        # Aggregate into allocation
        sector_list = list(signals_per_sector.keys())
        n_sectors = len(sector_list)

        if n_sectors == 0:
            return {"error": "No sector data available"}

        predictions = np.array([
            signals_per_sector[s]["prediction"] for s in sector_list
        ])

        # Get RL allocation
        dummy_state = np.zeros(self.ensemble.rl_agent.policy.backbone[0].in_features)
        rl_alloc, _, _ = self.ensemble.rl_agent.select_action(
            dummy_state, deterministic=True
        )
        rl_alloc = rl_alloc[:n_sectors]

        # Generate final signals
        signal_result = self.signal_gen.generate(
            predictions, rl_alloc
        )

        # Portfolio optimization
        weights = self.portfolio_opt.optimize(
            signal_result["signal"],
            pd.DataFrame(),  # Would use real returns data
            method="signal_weighted",
        )

        # Risk management
        pv = portfolio_value or self.config["backtest"]["initial_capital"]
        weights, risk_report = self.risk_mgr.check_and_adjust(weights, pv)

        return {
            "sector_allocations": dict(zip(sector_list, weights[:n_sectors].tolist())),
            "signals": signal_result,
            "risk_report": risk_report,
            "predictions": dict(zip(sector_list, predictions.tolist())),
            "confidence": signal_result["confidence"],
        }
