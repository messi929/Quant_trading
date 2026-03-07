"""Real-time inference pipeline for generating trading signals."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from data.collector import MarketDataCollector, DataCollector
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
from utils.ticker_utils import is_domestic


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

        # For live inference, only seq_length days needed — lower threshold
        self.processor = DataProcessor(min_history_days=self.config["data"]["sequence_length"])
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
        import torch as _torch
        vae_cfg = self.model_config["vae"]
        tf_cfg = self.model_config["transformer"]
        gan_cfg = self.model_config["gan"]
        rl_cfg = self.model_config["rl"]
        seq_length = self.config["data"]["sequence_length"]

        # Detect actual n_features from checkpoint to avoid shape mismatch
        # VAE encoder.network.0.weight shape = [hidden_dim, seq_length * n_features]
        if Path(path).exists():
            ckpt = _torch.load(path, map_location="cpu", weights_only=False)
            flat_dim = ckpt["vae"]["encoder.network.0.weight"].shape[1]
            n_features = flat_dim // seq_length
            logger.info(f"Checkpoint n_features detected: {n_features} (flat_dim={flat_dim})")
        else:
            n_features = 34  # fallback
            ckpt = None

        vae = MarketVAE(
            input_dim=n_features,
            latent_dim=vae_cfg["latent_dim"],
            seq_length=seq_length,
        )

        transformer = TemporalTransformer(
            input_dim=n_features,
            d_model=tf_cfg["d_model"],
            n_heads=tf_cfg["n_heads"],
            n_layers=tf_cfg["n_encoder_layers"],
            d_ff=tf_cfg["d_ff"],
            n_sectors=tf_cfg["n_sectors"],
            sector_embed_dim=tf_cfg.get("sector_embed_dim", 32),
        )

        gan = ConditionalWGAN(
            noise_dim=gan_cfg["noise_dim"],
            hidden_dims=gan_cfg.get("generator_hidden_dims", [64, 128, 64]),
            output_dim=n_features,
            seq_length=gan_cfg["sequence_length"],
        )

        # Detect RL state_dim from checkpoint (first linear layer input size)
        if ckpt is not None:
            rl_first_w = next(
                v for k, v in ckpt["rl_policy"].items() if "weight" in k
            )
            rl_state_dim = rl_first_w.shape[1]
        else:
            rl_state_dim = vae_cfg["latent_dim"] + tf_cfg["d_model"]
        agent = PPOAgent(
            state_dim=rl_state_dim,
            n_sectors=rl_cfg["n_sectors"],
            hidden_dims=rl_cfg["hidden_dims"],
            device=self.dm.device,
        )

        ensemble = ModelEnsemble(
            vae, transformer, gan, agent,
            device=self.dm.device,
        )

        if ckpt is not None:
            ensemble.load_ensemble(path)
            logger.info(f"Loaded ensemble from {path}")
        else:
            logger.warning(f"Ensemble not found at {path}, using untrained models")

        return ensemble

    def _build_sector_id_map(self) -> dict:
        """Load training-time sector→id mapping from sectors.yaml."""
        # Try to find sectors.yaml relative to config_path
        config_dir = Path(self.config.get("root_dir", "."))
        sectors_path = config_dir / "config" / "sectors.yaml"
        if not sectors_path.exists():
            sectors_path = Path("config/sectors.yaml")
        if not sectors_path.exists():
            logger.warning("sectors.yaml not found, using default order")
            return {}
        with open(sectors_path, "r", encoding="utf-8") as f:
            sectors_cfg = yaml.safe_load(f)
        return {
            name: idx
            for idx, name in enumerate(sectors_cfg["sectors"].keys())
        }

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
            # step_collect이 이미 저장한 processed_data를 재사용 (pykrx 재호출 방지)
            processed_path = Path("data/processed/processed_data.parquet")
            if processed_path.exists():
                market_data = pd.read_parquet(processed_path)
                logger.info(f"Loaded market data from parquet: {len(market_data):,} rows, "
                            f"{market_data['ticker'].nunique()} tickers")
            else:
                logger.warning("processed_data.parquet 없음 → pykrx 재수집 (느림)")
                collector = MarketDataCollector(history_years=1)
                market_data_dict = collector.collect_all(save=False)
                market_data = pd.concat(market_data_dict.values(), ignore_index=True)

        # Process
        df = self.processor.process(market_data)
        df = self.feature_eng.compute_all(df)
        feature_cols = self.feature_eng.get_feature_names()
        df = self.processor.normalize(df, feature_cols, fit=True)

        # Assign sector column from training data mapping (avoid slow API calls)
        if "sector" not in df.columns:
            processed_path = Path("data/processed/processed_data.parquet")
            if processed_path.exists():
                ticker_sector = (
                    pd.read_parquet(processed_path, columns=["ticker", "sector"])
                    .drop_duplicates("ticker")
                    .set_index("ticker")["sector"]
                )
                df["sector"] = df["ticker"].map(ticker_sector).fillna("unknown")
                logger.info(f"Sector assigned from training data: {df['sector'].nunique()} sectors")
            else:
                df["sector"] = "unknown"
                logger.warning("No sector mapping found, all tickers assigned to 'unknown'")

        # Build training-time sector ID map (fixes sector_id always=0 bug)
        sector_id_map = self._build_sector_id_map()

        # Per-ticker inference, averaged to sector level
        # (fixes mixed-ticker rows bug: sector_data has N_tickers×N_days rows)
        signals_per_sector = {}

        for sector in df["sector"].unique():
            if sector == "unknown":
                continue
            sector_df = df[df["sector"] == sector]
            tickers_in_sector = sector_df["ticker"].unique()

            train_sector_id = sector_id_map.get(sector, 0)
            sector_id_tensor = torch.tensor(
                [train_sector_id], dtype=torch.long
            ).to(self.dm.device)

            ticker_scores = {}  # ticker -> prediction score
            for ticker in tickers_in_sector:
                tkr_df = sector_df[sector_df["ticker"] == ticker].sort_values("date")
                if len(tkr_df) < self.seq_length:
                    continue

                window = tkr_df[feature_cols].values[-self.seq_length:]
                features = torch.FloatTensor(window).unsqueeze(0).to(self.dm.device)

                with torch.no_grad():
                    model_signals = self.ensemble.get_signals(features, sector_id_tensor)
                    ticker_scores[ticker] = model_signals["prediction"].cpu().item()

            if ticker_scores:
                # Rank tickers by model score, keep top N with positive signal only
                top_n = 3
                ranked = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
                top_tickers = [
                    {
                        "ticker": t,
                        "score": s,
                        "market": "domestic" if is_domestic(t) else "overseas",
                    }
                    for t, s in ranked[:top_n]
                    if s > 0  # 상승 신호 있는 종목만
                ]
                signals_per_sector[sector] = {
                    "prediction": float(np.mean(list(ticker_scores.values()))),
                    "n_tickers": len(ticker_scores),
                    "top_tickers": top_tickers,
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

        # Per-sector top tickers for individual stock ordering
        sector_top_tickers = {
            s: signals_per_sector[s].get("top_tickers", [])
            for s in sector_list
        }

        return {
            "sector_allocations": dict(zip(sector_list, weights[:n_sectors].tolist())),
            "sector_top_tickers": sector_top_tickers,
            "signals": signal_result,
            "risk_report": risk_report,
            "predictions": dict(zip(sector_list, predictions.tolist())),
            "confidence": signal_result["confidence"],
        }
