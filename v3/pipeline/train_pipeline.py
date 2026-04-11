"""E2E training pipeline: load data → train → evaluate → gate check."""

from __future__ import annotations

from loguru import logger

from v3.config.schema import V3Config
from v3.data.dataset import create_dataloaders
from v3.data.normalizer import FeatureNormalizer
from v3.model.vol_transformer import VolTransformer
from v3.model.trainer import VolTrainer
from v3.utils.device import DeviceManager, set_seed
from v3.utils.storage import StorageManager


class TrainPipeline:
    """End-to-end V3 training pipeline."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.storage = StorageManager(base_dir="v3")

    def run(self) -> dict:
        """Run full training pipeline.

        Returns:
            {"gate_pass": bool, "best_metrics": dict, "test_metrics": dict}
        """
        logger.info("=" * 60)
        logger.info("V3 Training Pipeline Start")
        logger.info("=" * 60)

        set_seed(self.cfg.training.seed)

        # Step 1: Load processed data
        logger.info("[1/5] Loading processed data...")
        df = self.storage.load_parquet("processed_data")
        feat_cfg = self.storage.load_json("feature_config.json")
        feature_cols = feat_cfg["feature_cols"]
        logger.info(f"  Data: {len(df)} rows, {df['ticker'].nunique()} tickers, "
                     f"{len(feature_cols)} features")

        # Step 2: Normalize features (fit on train split only)
        logger.info("[2/5] Normalizing features...")
        dates = sorted(df["date"].unique())
        train_end = int(len(dates) * self.cfg.data.train_ratio)
        train_dates = set(dates[:train_end])
        train_df = df[df["date"].isin(train_dates)]

        normalizer = FeatureNormalizer(clip_value=self.cfg.data.normalize_clip)
        normalizer.fit(train_df, feature_cols)
        df = normalizer.transform(df, feature_cols)
        normalizer.save(self.cfg.paths.normalizer_stats)

        # Step 3: Create dataloaders
        logger.info("[3/5] Creating dataloaders...")
        train_loader, val_loader, test_loader, test_df = create_dataloaders(
            df=df,
            feature_cols=feature_cols,
            target_col="vol_target",
            seq_len=self.cfg.data.sequence_length,
            batch_size=self.cfg.training.batch_size,
            train_ratio=self.cfg.data.train_ratio,
            val_ratio=self.cfg.data.val_ratio,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
        )
        logger.info(f"  Train: {len(train_loader.dataset)} samples, "
                     f"Val: {len(val_loader.dataset)}, "
                     f"Test: {len(test_loader.dataset)}")

        # Step 4: Train
        logger.info("[4/5] Training VolTransformer...")
        dm = DeviceManager(
            memory_fraction=self.cfg.gpu.memory_fraction,
            compile_model=self.cfg.gpu.compile_model,
        )

        model = VolTransformer(
            input_dim=len(feature_cols),
            d_model=self.cfg.model.d_model,
            n_heads=self.cfg.model.n_heads,
            n_layers=self.cfg.model.n_layers,
            d_ff=self.cfg.model.d_ff,
            dropout=self.cfg.model.dropout,
            max_seq_length=self.cfg.model.max_seq_length,
            use_confidence_head=self.cfg.model.use_confidence_head,
        )
        logger.info(f"  Model: {model.count_parameters():,} parameters")

        trainer = VolTrainer(
            model=model,
            device_manager=dm,
            learning_rate=self.cfg.training.learning_rate,
            warmup_steps=self.cfg.training.warmup_steps,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            max_grad_norm=self.cfg.training.max_grad_norm,
            weight_decay=self.cfg.training.weight_decay,
            ranking_loss_weight=self.cfg.training.ranking_loss_weight,
            confidence_loss_weight=self.cfg.training.confidence_loss_weight,
            min_vol_ic=self.cfg.gates.min_vol_ic,
            min_vol_rank_ic=self.cfg.gates.min_vol_rank_ic,
            min_dir_acc=self.cfg.gates.min_dir_acc,
        )

        train_result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.cfg.training.epochs,
            patience=self.cfg.training.early_stopping_patience,
            save_name="vol_transformer",
        )

        # Step 5: Test evaluation
        logger.info("[5/5] Test set evaluation...")
        test_metrics = trainer.evaluate_test(test_loader)

        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"  Val Gate:  {'PASS' if train_result['gate_pass'] else 'FAIL'}")
        logger.info(f"  Test Gate: {'PASS' if test_metrics['gate_pass'] else 'FAIL'}")
        logger.info(f"  Best Val Loss: {train_result['best_val_loss']:.4f}")
        logger.info("=" * 60)

        dm.clear_memory()

        return {
            "gate_pass": train_result["gate_pass"] and test_metrics["gate_pass"],
            "best_metrics": train_result["best_metrics"],
            "test_metrics": test_metrics,
        }
