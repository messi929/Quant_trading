"""Training pipeline (v2.3) — single model with deployment gates.

Flow:
  1. Load & process data
  2. Feature selection (remove redundant, keep 30~35)
  3. Fit normalizer on training split
  4. Create dataloaders (z-score normalized)
  5. Train AlphaTransformer (ListMLE + Huber)
  6. GATE: dir_acc > 54.5%, rank_ic > 0.10
  7. Save model + normalizer + feature_cols

v2.3 changes:
  - Feature selection: correlation-based redundancy removal (71→30~35)
  - ListMLE ranking loss (from alpha_trainer)
  - Tightened deployment gates
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config.config_loader import load_config
from data.dataset import create_dataloaders
from data.feature_engineer import FeatureEngineer
from models.alpha_model import AlphaTransformer
from models.alpha_trainer import AlphaTrainer
from utils.device import DeviceManager


def select_features(
    df: pd.DataFrame,
    candidates: list[str],
    target_col: str = "return_1d",
    max_features: int = 35,
    max_corr: float = 0.85,
) -> list[str]:
    """Select top features by removing redundancy.

    1. Compute absolute correlation of each feature with target
    2. Sort by relevance (high |corr| first)
    3. Greedily add features, skip if |corr| > max_corr with any already selected

    Args:
        df: DataFrame with features
        candidates: Candidate feature column names
        target_col: Target column for relevance scoring
        max_features: Maximum features to keep
        max_corr: Maximum inter-feature correlation allowed

    Returns:
        Selected feature column names (ordered by relevance)
    """
    # Filter to available columns
    available = [c for c in candidates if c in df.columns]
    if not available:
        return candidates[:max_features]

    # Sample for speed (correlation on full dataset is expensive)
    sample = df[available + [target_col]].dropna().sample(
        min(100_000, len(df)), random_state=42
    )

    # Relevance: absolute correlation with target
    relevance = sample[available].corrwith(sample[target_col]).abs().fillna(0)
    ranked = relevance.sort_values(ascending=False).index.tolist()

    # Greedy selection with redundancy filter
    selected = []
    # Deduplicate column names before correlation
    deduped = sample[available].loc[:, ~sample[available].columns.duplicated()]
    corr_matrix = deduped.corr().abs()

    for feat in ranked:
        if len(selected) >= max_features:
            break
        if feat not in corr_matrix.columns:
            continue

        # Check correlation with already selected features
        redundant = False
        for sel in selected:
            if sel not in corr_matrix.columns:
                continue
            val = corr_matrix.loc[feat, sel]
            if hasattr(val, 'item'):
                val = val.item()
            if float(val) > max_corr:
                redundant = True
                break

        if not redundant:
            selected.append(feat)

    logger.info(
        f"Feature selection: {len(candidates)} → {len(selected)} "
        f"(max_corr={max_corr}, max_features={max_features})"
    )
    logger.info(f"Top 10 features: {selected[:10]}")

    return selected


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

        # ── Step 1b: Merge flow data (if available) ──
        df = self._merge_flow_data(df)

        # ── Step 2: Feature columns (with selection) ──
        meta_cols = [
            "ticker", "date", "open", "high", "low", "close", "volume",
            "sector", "market", "market_cap", "market_return",
            "market_volatility", "market_breadth", "market_momentum",
            "relative_return",
            # Flow raw columns (features are foreign_net_ratio, inst_net_ratio, etc.)
            "foreign_net_buy", "inst_net_buy", "trading_value",
            # Sector raw column (features are sector_relative_return, etc.)
            "sector_return_1d",
        ]
        all_feature_cols = [
            c for c in df.columns
            if c not in meta_cols and df[c].dtype in ["float64", "float32", "int64"]
        ]
        logger.info(f"Candidate features: {len(all_feature_cols)} columns")

        # Feature selection: remove redundancy
        # max_features=45 to accommodate flow features alongside OHLCV features
        max_features = 45 if "foreign_net_ratio" in df.columns else 35
        feature_cols = select_features(
            df, all_feature_cols,
            target_col="return_1d",
            max_features=max_features,
            max_corr=0.85,
        )

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

        # ── Step 8: Walk-forward stability check ──
        logger.info("Step 6: Walk-forward stability check")
        wf_result = self._walk_forward_check(
            df, feature_cols, sector_map, cfg_data, cfg_train
        )

        # ── Step 9: Summary ──
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE V2.3 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Val gate pass:  {train_result['gate_pass']}")
        logger.info(f"  Test gate pass: {test_metrics['gate_pass']}")
        logger.info(f"  Best val loss:  {train_result['best_val_loss']:.4f}")
        logger.info(f"  Test dir_acc:   {test_metrics['dir_acc']:.4f}")
        logger.info(f"  Test rank_ic:   {test_metrics['rank_ic']:.4f}")
        if wf_result:
            logger.info(
                f"  Walk-forward:   {wf_result['n_profitable']}/{wf_result['n_folds']} "
                f"profitable folds, mean IC={wf_result['mean_ic']:.4f}"
            )

        if test_metrics["gate_pass"]:
            logger.info("  STATUS: READY FOR DEPLOYMENT")
        else:
            logger.warning("  STATUS: NOT READY — improve model before deployment")

        logger.info("=" * 60)

        return {
            "train_result": train_result,
            "test_metrics": test_metrics,
            "walk_forward": wf_result,
            "feature_cols": feature_cols,
            "n_params": n_params,
        }

    def _merge_flow_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge institutional/foreign flow data if flow_data.parquet exists.

        Flow data is collected separately (slow: Naver Finance scraping).
        If available, merge on (date, ticker) and compute flow features.
        NASDAQ tickers get flow=0 (no KRX flow data).
        """
        flow_path = Path(self.cfg["paths"]["raw_data"]) / "flow_data.parquet"
        if not flow_path.exists():
            logger.info("No flow_data.parquet found — training without flow features")
            return df

        logger.info(f"Merging flow data from {flow_path}")
        flow_df = pd.read_parquet(flow_path)
        flow_df["date"] = pd.to_datetime(flow_df["date"])

        # Ensure df date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        n_before = len(df)
        df = df.merge(
            flow_df[["date", "ticker", "foreign_net_buy", "inst_net_buy"]],
            on=["date", "ticker"],
            how="left",
        )

        # NASDAQ and unmatched tickers: fill NaN with 0 (no flow signal)
        df["foreign_net_buy"] = df["foreign_net_buy"].fillna(0.0)
        df["inst_net_buy"] = df["inst_net_buy"].fillna(0.0)

        n_matched = (df["foreign_net_buy"] != 0).sum()
        logger.info(
            f"Flow data merged: {n_matched:,}/{len(df):,} rows with flow data "
            f"({n_matched/len(df)*100:.1f}%)"
        )

        # Compute flow features
        try:
            from data.flow_data import FlowFeatureEngineer
            flow_fe = FlowFeatureEngineer()
            df = flow_fe.compute_flow_features(df)
        except Exception as e:
            logger.warning(f"Flow feature computation failed: {e}")

        return df

    def _walk_forward_check(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        sector_map: dict,
        cfg_data: dict,
        cfg_train: dict,
    ) -> dict | None:
        """Walk-forward validation: retrain on rolling windows.

        Uses config walk_forward params (train_days=252, test_days=63, step=63).
        Trains a fresh model per fold, evaluates Rank IC on test window.
        """
        wf_cfg = self.cfg.get("backtest", {}).get("walk_forward", {})
        train_days = wf_cfg.get("train_days", 252)
        test_days = wf_cfg.get("test_days", 63)
        step_days = wf_cfg.get("step_days", 63)

        dates = sorted(df["date"].unique())
        n_dates = len(dates)
        min_required = train_days + test_days

        if n_dates < min_required:
            logger.warning(
                f"Walk-forward skipped: need {min_required} dates, have {n_dates}"
            )
            return None

        fold_ics = []
        fold_accs = []
        n_folds = 0

        # Walk forward from the middle of the dataset
        start_idx = max(0, n_dates - train_days - test_days * 4)
        while start_idx + train_days + test_days <= n_dates:
            train_end_date = dates[start_idx + train_days]
            test_end_date = dates[min(start_idx + train_days + test_days, n_dates) - 1]

            fold_train = df[
                (df["date"] >= dates[start_idx]) & (df["date"] < train_end_date)
            ].copy()
            fold_test = df[
                (df["date"] >= train_end_date) & (df["date"] <= test_end_date)
            ].copy()

            if len(fold_train) < 1000 or len(fold_test) < 200:
                start_idx += step_days
                continue

            try:
                from data.normalizer import FeatureNormalizer
                from data.dataset import MarketSequenceDataset
                from torch.utils.data import DataLoader

                norm = FeatureNormalizer()
                norm.fit(fold_train, feature_cols)
                fold_train = norm.transform(fold_train, feature_cols)
                fold_test = norm.transform(fold_test, feature_cols)

                test_ds = MarketSequenceDataset(
                    fold_test, feature_cols,
                    cfg_data["sequence_length"],
                    cfg_data["prediction_horizon"],
                    sector_map,
                    cross_sectional_target=cfg_data["cross_sectional_target"],
                )
                test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

                # Quick-train a model (reduced epochs for speed)
                cfg_model = self.cfg["model"]
                fold_model = AlphaTransformer(
                    input_dim=len(feature_cols),
                    d_model=cfg_model["d_model"],
                    n_heads=cfg_model["n_heads"],
                    n_layers=cfg_model["n_encoder_layers"],
                    d_ff=cfg_model["d_ff"],
                    dropout=cfg_model["dropout"],
                    max_seq_length=cfg_model["max_seq_length"],
                    use_confidence_head=cfg_model["use_confidence_head"],
                )

                train_ds = MarketSequenceDataset(
                    fold_train, feature_cols,
                    cfg_data["sequence_length"],
                    cfg_data["prediction_horizon"],
                    sector_map,
                    cross_sectional_target=cfg_data["cross_sectional_target"],
                )
                train_loader = DataLoader(
                    train_ds, batch_size=256, shuffle=True, drop_last=True
                )

                fold_trainer = AlphaTrainer(
                    model=fold_model,
                    device_manager=self.dm,
                    learning_rate=cfg_train["learning_rate"],
                    warmup_steps=200,
                    ranking_loss_weight=cfg_train["ranking_loss_weight"],
                    confidence_loss_weight=cfg_train["confidence_loss_weight"],
                    min_dir_acc=cfg_train["min_dir_acc"],
                    min_rank_ic=cfg_train["min_rank_ic"],
                )

                # Quick train (15 epochs, no early stopping)
                from models.alpha_trainer import cosine_warmup_scheduler
                total_steps = 15 * len(train_loader)
                scheduler = cosine_warmup_scheduler(
                    fold_trainer.optimizer, 200, total_steps
                )
                for _ in range(15):
                    fold_trainer.train_epoch(train_loader, scheduler)

                metrics = fold_trainer.validate(test_loader)
                fold_ics.append(metrics["rank_ic"])
                fold_accs.append(metrics["dir_acc"])
                n_folds += 1

                logger.info(
                    f"  WF fold {n_folds}: "
                    f"train={dates[start_idx]:%Y-%m-%d}~{train_end_date:%Y-%m-%d}, "
                    f"test=~{test_end_date:%Y-%m-%d} | "
                    f"IC={metrics['rank_ic']:.4f}, Acc={metrics['dir_acc']:.4f}"
                )

                self.dm.clear_memory()

            except Exception as e:
                logger.warning(f"WF fold failed: {e}")

            start_idx += step_days

        if not fold_ics:
            return None

        result = {
            "n_folds": n_folds,
            "fold_ics": fold_ics,
            "fold_accs": fold_accs,
            "mean_ic": float(np.mean(fold_ics)),
            "std_ic": float(np.std(fold_ics)),
            "mean_acc": float(np.mean(fold_accs)),
            "n_profitable": sum(1 for ic in fold_ics if ic > 0),
            "stability": float(np.std(fold_ics) / (abs(np.mean(fold_ics)) + 1e-8)),
        }

        logger.info(
            f"Walk-forward: {result['n_profitable']}/{n_folds} profitable, "
            f"mean IC={result['mean_ic']:.4f} ± {result['std_ic']:.4f}, "
            f"stability={result['stability']:.2f}"
        )

        return result
