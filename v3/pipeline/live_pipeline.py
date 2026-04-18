"""E2E live trading pipeline: data → inference → signal → execute."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from v3.config.schema import V3Config, load_config
from v3.data.collector import OHLCVCollector
from v3.data.macro_collector import MacroCollector
from v3.data.macro_features import MacroFeatureEngineer
from v3.data.universe import Universe
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.data.dart_client import DartClient
from v3.model.vol_transformer import VolTransformer
from v3.model.inference import VolInference
from v3.rules.direction import DirectionEngine
from v3.rules.entry import EntryFilter
from v3.strategy.signal import SignalGenerator
from v3.strategy.sizing import VolTargetSizer
from v3.strategy.regime import RegimeDetector
from v3.strategy.regime_cross_asset import CrossAssetRegimeDetector
from v3.execution.executor import TradingExecutor
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


class LivePipeline:
    """Full live trading pipeline for V3."""

    def __init__(self, cfg: V3Config | None = None):
        self.cfg = cfg or load_config()
        self.storage = StorageManager(base_dir="v3")

        # Data
        self.universe = Universe(
            markets=self.cfg.data.markets,
            kospi200_only=self.cfg.data.kospi200_only,
        )
        self.collector = OHLCVCollector(
            save_dir=self.cfg.paths.raw_data,
            history_years=self.cfg.data.history_years,
        )
        self.feature_eng = VolFeatureEngineer()
        self.normalizer = FeatureNormalizer.load(self.cfg.paths.normalizer_stats)
        self.dart = DartClient()

        # Model
        dm = DeviceManager(compile_model=False)
        feat_cfg = self.storage.load_json("feature_config.json")
        self.feature_cols = feat_cfg["feature_cols"]

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
        self.inference = VolInference(model, self.normalizer, self.feature_cols, dm)

        # Strategy
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

        # Regime engine: cross_asset(신규) or single_asset(legacy)
        if self.cfg.regime.engine == "cross_asset":
            self.macro_collector = MacroCollector(
                save_dir=self.cfg.paths.raw_data,
                history_years=self.cfg.regime.macro_history_years,
            )
            self.macro_features = MacroFeatureEngineer()
            self.regime_detector = CrossAssetRegimeDetector(
                hysteresis_days=self.cfg.regime.hysteresis_days,
            )
            logger.info("Regime engine: cross_asset (multi-signal composite)")
        else:
            self.macro_collector = None
            self.macro_features = None
            self.regime_detector = RegimeDetector()
            logger.info("Regime engine: single_asset (legacy)")

        # Execution
        self.executor = TradingExecutor(self.cfg)

    def collect_data(self) -> pd.DataFrame:
        """Collect latest OHLCV data (incremental)."""
        logger.info("Collecting data (incremental 10 days)...")
        self.universe.build()

        existing = self.collector.load_existing()
        new_data = self.collector.collect(self.universe, incremental_days=10)

        if existing is not None and not new_data.empty:
            df = self.collector.merge_incremental(existing, new_data)
        elif existing is not None:
            df = existing
        else:
            df = new_data

        # Feature engineering
        df = self.feature_eng.compute_all(df)
        return df

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """Generate trading signal from latest data."""
        logger.info("Generating signal...")

        # Inference
        vol_scores = self.inference.predict(df)

        # Regime detection
        regime_state = self._detect_regime(df)
        if self.cfg.regime.engine == "cross_asset":
            logger.info(f"Regime: {regime_state.regime} score={regime_state.score:.2f} "
                         f"scale={regime_state.scale_factor:.2f} "
                         f"thresh_mult={regime_state.threshold_multiplier:.2f} "
                         f"contribs={regime_state.contributions}")
        else:
            logger.info(f"Regime: {regime_state.regime} (momentum={regime_state.momentum:.2%}, "
                         f"vol={regime_state.volatility:.2%})")

        # DART events
        event_scores = {}
        if self.dart.is_available:
            event_scores = self.dart.get_event_signals(days_back=3)

        # Monthly trade count
        today = datetime.now().strftime("%Y-%m-%d")
        monthly = self.executor.positions.monthly_trade_count(today)

        # Generate signal
        signal = self.signal_gen.generate(
            vol_scores=vol_scores,
            full_data=df,
            regime_state=regime_state,
            event_scores=event_scores,
            current_positions=self.executor.positions.count(),
            monthly_trades=monthly,
            market_cost=self.cfg.trading.costs.us_roundtrip,
        )

        logger.info(f"Signal: {signal.action} — {len(signal.positions)} positions, "
                     f"regime={signal.regime}, cash={signal.cash_weight:.0%}")

        return {
            "action": signal.action,
            "positions": signal.positions,
            "regime": signal.regime,
            "cash_weight": signal.cash_weight,
        }

    def _detect_regime(self, ohlcv: pd.DataFrame):
        """Dispatch to single_asset or cross_asset regime engine."""
        if self.cfg.regime.engine == "cross_asset" and self.macro_collector is not None:
            # Collect latest macro (incremental 90d)
            existing = self.macro_collector.load()
            if existing is None or len(existing) < 60:
                macro = self.macro_collector.collect()
            else:
                macro = self.macro_collector.collect(incremental_days=90)

            # Compute features + percentiles
            feats = self.macro_features.compute(macro, ohlcv)
            pctl = self.macro_features.compute_percentiles(feats)

            # Warm up hysteresis by replaying last 5 days (first startup only)
            if self.regime_detector._days_in_regime == 0 and len(pctl) >= 5:
                for i in range(-5, 0):
                    self.regime_detector.detect(pctl.iloc[: len(pctl) + i + 1 if i < -1 else None])
            return self.regime_detector.detect(pctl)

        # Legacy single-asset fallback
        market_data = ohlcv.groupby("date").agg(
            close=("close", "mean")
        ).reset_index().sort_values("date")
        return self.regime_detector.detect(market_data)

    def execute(self, signal: dict) -> list[dict]:
        """Execute trading signal."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.executor.execute_entry(signal, today)

    def monitor(self) -> list[dict]:
        """Monitor open positions for exits."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.executor.monitor_positions(today)

    def run_session(self) -> dict:
        """Run a complete trading session.

        Steps: collect → signal → execute → monitor.
        """
        logger.info("=" * 60)
        logger.info(f"V3 Trading Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        # 1. Collect data
        df = self.collect_data()

        # 2. Generate signal
        signal = self.generate_signal(df)

        # 3. Execute entry
        entries = self.execute(signal)
        logger.info(f"Entries: {len(entries)}")

        # 4. Monitor positions
        exits = self.monitor()
        logger.info(f"Exits: {len(exits)}")

        # Summary
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "signal": signal["action"],
            "regime": signal["regime"],
            "entries": len(entries),
            "exits": len(exits),
            "open_positions": self.executor.positions.count(),
        }

        logger.info(f"Session complete: {summary}")
        return summary
