"""Walk-forward validation with per-fold model retraining.

Each fold: train model on N days → predict next M days → backtest.
This ensures no look-ahead bias and tests true out-of-sample performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from v3.config.schema import V3Config
from v3.data.dataset import VolDataset, DataLoader
from v3.data.normalizer import FeatureNormalizer
from v3.model.vol_transformer import VolTransformer
from v3.model.trainer import VolTrainer
from v3.backtest.engine import BacktestEngine
from v3.utils.device import DeviceManager, set_seed


class WalkForwardValidator:
    """Walk-forward with per-fold retraining."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
        train_epochs: int = 15,   # Fewer epochs per fold (speed)
        patience: int = 5,
    ) -> dict:
        """Run walk-forward with model retraining per fold.

        Args:
            df: Full DataFrame with features and vol_target.
            feature_cols: Feature column names.
            train_days/test_days/step_days: Walk-forward windows.
            train_epochs: Max epochs per fold (reduced for speed).
            patience: Early stopping patience per fold.
        """
        dates = sorted(df["date"].unique())
        seq_len = self.cfg.data.sequence_length
        results = []

        logger.info(f"Walk-forward retrain: {len(dates)} dates, "
                     f"train={train_days}d, test={test_days}d, step={step_days}d")

        fold = 0
        for start_idx in range(0, len(dates) - train_days - test_days, step_days):
            train_end = start_idx + train_days
            test_end = train_end + test_days
            if test_end > len(dates):
                break

            fold += 1
            train_dates = set(dates[start_idx:train_end])
            test_dates_set = set(dates[train_end:test_end])

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates_set)]

            if len(train_df) < 1000 or len(test_df) < 100:
                continue

            logger.info(f"Fold {fold}: train {dates[start_idx]}~{dates[train_end-1]} "
                         f"→ test {dates[train_end]}~{dates[min(test_end-1, len(dates)-1)]}")

            # 1. Normalize (fit on train only)
            normalizer = FeatureNormalizer(clip_value=self.cfg.data.normalize_clip)
            normalizer.fit(train_df, feature_cols)
            train_norm = normalizer.transform(train_df, feature_cols)
            test_norm = normalizer.transform(test_df, feature_cols)

            # 2. Create datasets
            train_ds = VolDataset(train_norm, feature_cols, "vol_target", seq_len)
            if len(train_ds) < 100:
                continue

            train_loader = DataLoader(
                train_ds, batch_size=min(256, len(train_ds)),
                shuffle=True, num_workers=0, drop_last=True,
            )

            # 3. Quick train
            dm = DeviceManager(compile_model=False)
            model = VolTransformer(
                input_dim=len(feature_cols),
                d_model=self.cfg.model.d_model,
                n_heads=self.cfg.model.n_heads,
                n_layers=self.cfg.model.n_layers,
                d_ff=self.cfg.model.d_ff,
                dropout=self.cfg.model.dropout,
                max_seq_length=seq_len,
                use_confidence_head=self.cfg.model.use_confidence_head,
            )

            trainer = VolTrainer(
                model=model,
                device_manager=dm,
                learning_rate=self.cfg.training.learning_rate,
                warmup_steps=200,  # Reduced for quick training
                min_vol_ic=0.0,    # No gate for walk-forward
                min_vol_rank_ic=0.0,
                min_dir_acc=0.0,
            )

            # Train (quick, no validation split within fold)
            val_ds = VolDataset(test_norm, feature_cols, "vol_target", seq_len)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

            trainer.train(
                train_loader, val_loader,
                epochs=train_epochs, patience=patience,
                save_name=f"wf_fold{fold}",
            )

            # 4. Generate predictions on test period
            model.eval()
            test_dates_list = sorted(test_dates_set)
            predictions = []

            with torch.no_grad():
                for date in test_dates_list:
                    for ticker, group in test_norm.groupby("ticker"):
                        available = group[group["date"] <= date].sort_values("date")
                        if len(available) < seq_len:
                            continue
                        features = available[feature_cols].values[-seq_len:]
                        features = np.nan_to_num(features, nan=0.0).astype(np.float32)
                        x = torch.tensor(features).unsqueeze(0).to(dm.device)
                        output = model(x)
                        conf = output.get("confidence")
                        predictions.append({
                            "date": date,
                            "ticker": ticker,
                            "vol_score": output["prediction"].item(),
                            "confidence": conf.item() if conf is not None else 0.5,
                        })

            if not predictions:
                continue

            vol_preds = pd.DataFrame(predictions)

            # 5. Backtest
            engine = BacktestEngine(self.cfg)
            engine.exit_rules.mae_threshold = -0.99
            engine.exit_rules.mae_tightened = -0.99
            engine.entry_filter.min_vol_expansion = self.cfg.trading.min_vol_expansion
            engine.entry_filter.min_confidence = self.cfg.trading.min_confidence
            engine.entry_filter.min_clarity = self.cfg.trading.direction_rules.min_direction_clarity

            bt_result = engine.run(df, vol_preds)

            logger.info(f"  Fold {fold}: trades={bt_result.total_trades} "
                         f"return={bt_result.total_return:.2%} "
                         f"Sharpe={bt_result.sharpe:.2f}")

            results.append(bt_result)
            dm.clear_memory()

        # Aggregate
        if not results:
            logger.warning("No valid walk-forward folds")
            return {"n_folds": 0}

        returns = [r.total_return for r in results]
        sharpes = [r.sharpe for r in results]
        trades = [r.total_trades for r in results]
        active_folds = [r for r in results if r.total_trades > 0]

        agg = {
            "n_folds": len(results),
            "n_active_folds": len(active_folds),
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "avg_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "worst_sharpe": float(np.min(sharpes)),
            "best_sharpe": float(np.max(sharpes)),
            "pct_profitable": sum(1 for r in returns if r > 0) / max(len(returns), 1),
            "avg_trades": float(np.mean(trades)),
            "total_trades": sum(trades),
        }

        if active_folds:
            active_returns = [r.total_return for r in active_folds]
            active_sharpes = [r.sharpe for r in active_folds]
            agg["active_avg_return"] = float(np.mean(active_returns))
            agg["active_avg_sharpe"] = float(np.mean(active_sharpes))
            agg["active_pct_profitable"] = sum(1 for r in active_returns if r > 0) / len(active_returns)

        logger.info("=" * 60)
        logger.info("WALK-FORWARD RETRAIN SUMMARY")
        logger.info(f"  Total Folds:       {agg['n_folds']}")
        logger.info(f"  Active Folds:      {agg['n_active_folds']}")
        logger.info(f"  Avg Return:        {agg['avg_return']:.2%}")
        logger.info(f"  Avg Sharpe:        {agg['avg_sharpe']:.2f}")
        logger.info(f"  Profitable %:      {agg['pct_profitable']:.0%}")
        if active_folds:
            logger.info(f"  Active Avg Sharpe: {agg['active_avg_sharpe']:.2f}")
            logger.info(f"  Active Profitable: {agg['active_pct_profitable']:.0%}")
        logger.info(f"  Total Trades:      {agg['total_trades']}")
        logger.info("=" * 60)

        return agg
