"""E2E data pipeline: collect → process → features → target → save."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from v3.config.schema import V3Config
from v3.data.universe import Universe
from v3.data.collector import OHLCVCollector
from v3.data.flow_collector import FlowDataCollector, FlowFeatureEngineer
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.utils.storage import StorageManager


class DataPipeline:
    """End-to-end data pipeline for V3 vol expansion trader."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.storage = StorageManager(base_dir="v3")
        self.universe = Universe(
            markets=cfg.data.markets,
            kospi200_only=cfg.data.kospi200_only,
        )
        self.collector = OHLCVCollector(
            save_dir=cfg.paths.raw_data,
            history_years=cfg.data.history_years,
        )
        self.feature_eng = VolFeatureEngineer()

    def run(self, incremental_days: int | None = None, collect_flow: bool = True) -> dict:
        """Run full data pipeline.

        Args:
            incremental_days: If set, only fetch last N days.
            collect_flow: Whether to collect flow data (slow, Naver scraping).

        Returns:
            {"df": processed DataFrame, "feature_cols": selected features, "stats": dict}
        """
        logger.info("=" * 60)
        logger.info("V3 Data Pipeline Start")
        logger.info("=" * 60)

        # Step 1: Build universe
        logger.info("[1/6] Building universe...")
        self.universe.build()
        logger.info(f"  Universe: {self.universe.count()} tickers")

        # Step 2: Collect OHLCV
        logger.info("[2/6] Collecting OHLCV data...")
        if incremental_days:
            existing = self.collector.load_existing()
            new_data = self.collector.collect(self.universe, incremental_days=incremental_days)
            if existing is not None and len(new_data) > 0:
                df = self.collector.merge_incremental(existing, new_data)
            elif existing is not None:
                df = existing
            else:
                df = new_data
        else:
            df = self.collector.collect(self.universe)

        if df.empty:
            logger.error("No data collected. Aborting.")
            return {"df": df, "feature_cols": [], "stats": {}}

        # Filter tickers with insufficient history
        min_days = self.cfg.data.min_history_days
        ticker_counts = df.groupby("ticker")["date"].count()
        valid_tickers = ticker_counts[ticker_counts >= min_days].index
        df = df[df["ticker"].isin(valid_tickers)]
        logger.info(f"  After min_history filter: {df['ticker'].nunique()} tickers, {len(df)} rows")

        # Step 3: Collect flow data (optional)
        has_flow = False
        if collect_flow:
            logger.info("[3/6] Collecting flow data...")
            try:
                krx_tickers = [t for t in df["ticker"].unique() if t.split(".")[0].isdigit()]
                if krx_tickers:
                    flow_collector = FlowDataCollector(save_dir=self.cfg.paths.raw_data)
                    start_date = df["date"].min().strftime("%Y-%m-%d")
                    flow_df = flow_collector.collect(krx_tickers, start_date=start_date)

                    if not flow_df.empty:
                        df = df.merge(
                            flow_df[["date", "ticker", "foreign_net_buy", "inst_net_buy"]],
                            on=["date", "ticker"],
                            how="left",
                        )
                        # Fill NaN flow data with 0
                        df["foreign_net_buy"] = df["foreign_net_buy"].fillna(0)
                        df["inst_net_buy"] = df["inst_net_buy"].fillna(0)

                        df = FlowFeatureEngineer.compute(df)
                        has_flow = True
                        logger.info(f"  Flow features added for {flow_df['ticker'].nunique()} tickers")
            except Exception as e:
                logger.warning(f"  Flow data collection failed: {e}")
        else:
            logger.info("[3/6] Skipping flow data collection")

        # Step 4: Feature engineering
        logger.info("[4/6] Computing features...")
        df = self.feature_eng.compute_all(df)

        # Step 5: Compute vol target
        logger.info("[5/6] Computing vol expansion target...")
        df = self.feature_eng.compute_vol_target(
            df,
            horizon=self.cfg.data.vol_prediction_horizon,
            baseline_window=self.cfg.data.vol_baseline_window,
        )

        # Step 6: Feature selection
        logger.info("[6/6] Selecting features...")
        feature_cols = self.feature_eng.select_features(
            df,
            target_col="vol_target",
            max_features=35,
            max_corr=0.85,
            include_flow=has_flow,
        )

        # Save processed data
        self.storage.save_parquet(df, "processed_data")
        self.storage.save_json(
            {"feature_cols": feature_cols, "has_flow": has_flow},
            "feature_config.json",
        )

        # Compute stats
        valid_targets = df["vol_target"].dropna()
        stats = {
            "total_rows": len(df),
            "n_tickers": df["ticker"].nunique(),
            "n_features": len(feature_cols),
            "has_flow": has_flow,
            "target_mean": float(valid_targets.mean()),
            "target_std": float(valid_targets.std()),
            "valid_samples": int(valid_targets.count()),
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}",
        }

        # Compute feature-target IC
        ic_values = {}
        valid = df[feature_cols + ["vol_target"]].dropna()
        for col in feature_cols:
            ic = valid[col].corr(valid["vol_target"], method="spearman")
            ic_values[col] = round(ic, 4)
        stats["feature_ic"] = ic_values
        top_ic = sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        logger.info(f"  Top 5 feature ICs: {top_ic}")

        logger.info("=" * 60)
        logger.info(f"Pipeline complete: {stats['n_tickers']} tickers, "
                     f"{stats['n_features']} features, {stats['valid_samples']} samples")
        logger.info("=" * 60)

        return {"df": df, "feature_cols": feature_cols, "stats": stats}
