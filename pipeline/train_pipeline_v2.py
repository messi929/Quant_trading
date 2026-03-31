"""Training pipeline (v2) — single model with deployment gates.

Flow:
  1. Load & process data
  2. Fit normalizer on training split
  3. Create dataloaders (z-score normalized, horizon=1)
  4. Train AlphaTransformer
  5. GATE: dir_acc > 52%, rank_ic > 0.10
  6. Save model + normalizer + feature_cols

Replaces v1 6-phase pipeline (VAE→Transformer→GAN→RL→Ensemble).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from config.config_loader import load_config
from data.dataset import create_dataloaders
from data.feature_engineer import FeatureEngineer
from models.alpha_model import AlphaTransformer
from models.alpha_trainer import AlphaTrainer
from utils.device import DeviceManager


class TrainingPipelineV2:
    """Single-model training pipeline with gates."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.dm = DeviceManager(compile_model=False)

    def run(self, data_path: str | None = None) -> dict:
        """Run full training pipeline.

        Args:
            data_path: Path to processed parquet. Default from config.

        Returns:
            Dict with training results, metrics, and gate status.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE V2 START")
        logger.info("=" * 60)

        # ── Step 1: Load data ──
        logger.info("Step 1: Loading data")
        if data_path is None:
            data_path = str(
                Path(self.cfg["paths"]["processed_data"]) / "processed_data.parquet"
            )
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

        # ── Step 2: Feature columns ──
        meta_cols = [
            "ticker", "date", "open", "high", "low", "close", "volume",
            "sector", "market", "market_cap", "market_return",
            "market_volatility", "market_breadth", "market_momentum",
            "relative_return",
        ]
        feature_cols = [
            c for c in df.columns
            if c not in meta_cols and df[c].dtype in ["float64", "float32", "int64"]
        ]
        logger.info(f"Features: {len(feature_cols)} columns")

        # Save feature cols
        cols_path = Path(self.cfg["paths"]["models"]) / "feature_cols.json"
        cols_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cols_path, "w") as f:
            json.dump(feature_cols, f)

        # ── Step 3: Sector map ──
        from live.sector_instruments import get_sector_order
        sector_map = {s: i for i, s in enumerate(get_sector_order())}

        # ── Step 4: Create dataloaders with normalization ──
        logger.info("Step 2: Creating dataloaders (z-score normalization)")
        cfg_data = self.cfg["data"]
        cfg_train = self.cfg["training"]

        train_loader, val_loader, test_loader, normalizer = create_dataloaders(
            df,
            feature_cols,
            sector_map,
            seq_length=cfg_data["sequence_length"],
            prediction_horizon=cfg_data["prediction_horizon"],
            batch_size=cfg_train["batch_size"],
            train_ratio=cfg_data["train_ratio"],
            val_ratio=cfg_data["val_ratio"],
            num_workers=cfg_train["num_workers"],
            pin_memory=cfg_train["pin_memory"],
            cross_sectional_target=cfg_data["cross_sectional_target"],
            normalize=cfg_data["normalize_features"],
            normalizer_save_path=self.cfg["paths"]["normalizer_stats"],
        )

        # ── Step 5: Create model ──
        logger.info("Step 3: Creating AlphaTransformer")
        cfg_model = self.cfg["model"]
        model = AlphaTransformer(
            input_dim=len(feature_cols),
            d_model=cfg_model["d_model"],
            n_heads=cfg_model["n_heads"],
            n_layers=cfg_model["n_encoder_layers"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model["dropout"],
            max_seq_length=cfg_model["max_seq_length"],
            use_confidence_head=cfg_model["use_confidence_head"],
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

        # ── Step 6: Train ──
        logger.info("Step 4: Training")
        trainer = AlphaTrainer(
            model=model,
            device_manager=self.dm,
            learning_rate=cfg_train["learning_rate"],
            warmup_steps=cfg_train["warmup_steps"],
            gradient_accumulation_steps=cfg_train["gradient_accumulation_steps"],
            max_grad_norm=cfg_train["max_grad_norm"],
            weight_decay=cfg_train["weight_decay"],
            ranking_loss_weight=cfg_train["ranking_loss_weight"],
            confidence_loss_weight=cfg_train["confidence_loss_weight"],
            min_dir_acc=cfg_train["min_dir_acc"],
            min_rank_ic=cfg_train["min_rank_ic"],
        )

        train_result = trainer.train(
            train_loader,
            val_loader,
            epochs=cfg_train["epochs"],
            patience=cfg_train["early_stopping_patience"],
            save_name="alpha_transformer",
        )

        # ── Step 7: Test evaluation ──
        logger.info("Step 5: Test set evaluation (out-of-sample)")
        test_metrics = trainer.evaluate_test(test_loader)

        # ── Step 8: Summary ──
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE V2 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Val gate pass:  {train_result['gate_pass']}")
        logger.info(f"  Test gate pass: {test_metrics['gate_pass']}")
        logger.info(f"  Best val loss:  {train_result['best_val_loss']:.4f}")
        logger.info(f"  Test dir_acc:   {test_metrics['dir_acc']:.4f}")
        logger.info(f"  Test rank_ic:   {test_metrics['rank_ic']:.4f}")

        if test_metrics["gate_pass"]:
            logger.info("  STATUS: READY FOR DEPLOYMENT")
        else:
            logger.warning("  STATUS: NOT READY — improve model before deployment")

        logger.info("=" * 60)

        return {
            "train_result": train_result,
            "test_metrics": test_metrics,
            "feature_cols": feature_cols,
            "n_params": n_params,
        }
