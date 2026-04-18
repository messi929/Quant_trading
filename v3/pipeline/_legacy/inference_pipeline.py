"""Standalone inference pipeline: load model → predict → signal."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from v3.config.schema import V3Config, load_config
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.model.vol_transformer import VolTransformer
from v3.model.inference import VolInference
from v3.rules.direction import DirectionEngine
from v3.rules.entry import EntryFilter
from v3.strategy.signal import SignalGenerator, TradeSignal
from v3.strategy.sizing import VolTargetSizer
from v3.strategy.regime import RegimeDetector
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


class InferencePipeline:
    """Generates signals from pre-loaded data without executing trades."""

    def __init__(self, cfg: V3Config | None = None):
        self.cfg = cfg or load_config()
        self.storage = StorageManager(base_dir="v3")

        # Load model
        dm = DeviceManager(compile_model=False)
        feat_cfg = self.storage.load_json("feature_config.json")
        self.feature_cols = feat_cfg["feature_cols"]

        normalizer = FeatureNormalizer.load(self.cfg.paths.normalizer_stats)
        checkpoint = self.storage.load_checkpoint("vol_transformer")
        model = VolTransformer(
            input_dim=len(self.feature_cols),
            d_model=self.cfg.model.d_model,
            n_heads=self.cfg.model.n_heads,
            n_layers=self.cfg.model.n_layers,
            d_ff=self.cfg.model.d_ff,
            dropout=self.cfg.model.dropout,
            max_seq_length=self.cfg.model.max_seq_length,
            use_confidence_head=self.cfg.model.use_confidence_head,
        )
        model.load_state_dict(checkpoint["state_dict"])
        self.inference = VolInference(model, normalizer, self.feature_cols, dm)

        # Strategy components
        direction_engine = DirectionEngine(
            momentum_window=self.cfg.trading.direction_rules.momentum_window,
            momentum_weight=self.cfg.trading.direction_rules.momentum_weight,
            flow_weight=self.cfg.trading.direction_rules.flow_weight,
            event_weight=self.cfg.trading.direction_rules.event_weight,
        )
        entry_filter = EntryFilter(
            min_direction_clarity=self.cfg.trading.direction_rules.min_direction_clarity,
            max_trades_per_month=self.cfg.trading.max_trades_per_month,
            max_positions=self.cfg.trading.max_positions,
            min_vol_expansion=self.cfg.trading.min_vol_expansion,
            min_confidence=self.cfg.trading.min_confidence,
        )
        sizer = VolTargetSizer(target_annual_vol=self.cfg.trading.target_annual_vol)
        self.signal_gen = SignalGenerator(direction_engine, entry_filter, sizer)
        self.regime_detector = RegimeDetector()

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from DataFrame with features."""
        vol_scores = self.inference.predict(df)

        market_data = df.groupby("date").agg(close=("close", "mean")).reset_index().sort_values("date")
        regime_state = self.regime_detector.detect(market_data)

        signal = self.signal_gen.generate(
            vol_scores=vol_scores,
            full_data=df,
            regime_state=regime_state,
        )

        return signal
