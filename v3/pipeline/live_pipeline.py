"""E2E live trading pipeline (Phase 2 S4 integration).

Single canonical path:
  OHLCV → features → VolTransformer inference → alpha_sources
  Macro → features → percentiles → RegimeDetectorV2 → Regime snapshot
  (alphas, conviction, regime, cost, state) → SignalGenerator → TradeSignal → execute
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from v3.config.schema import V3Config, load_config
from v3.data.collector import OHLCVCollector
from v3.data.dart_client import DartClient
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.macro_collector import MacroCollector
from v3.data.macro_features import MacroFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.data.universe import Universe
from v3.execution.executor import TradingExecutor
from v3.model.inference import VolInference
from v3.model.vol_transformer import VolTransformer
from v3.rules.entry import EntryFilter, OperationalState
from v3.strategy.alpha_sources import DEFAULT_CONVICTION, DEFAULT_DIRECTIONAL
from v3.strategy.opportunity import OpportunityScorer
from v3.strategy.regime_v2 import RegimeDetectorV2
from v3.strategy.signal import SignalGenerator
from v3.strategy.sizing import VolTargetSizer
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


class LivePipeline:
    """Full live trading pipeline for V3 Phase 2."""

    def __init__(self, cfg: V3Config | None = None):
        self.cfg = cfg or load_config()
        self.storage = StorageManager(base_dir="v3")

        # Data layer
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

        # Macro + regime
        self.macro_collector = MacroCollector(
            save_dir=self.cfg.paths.raw_data,
            history_years=self.cfg.regime.macro_history_years,
        )
        self.macro_fe = MacroFeatureEngineer()
        self.regime_detector = RegimeDetectorV2(
            weights_path="v3/config/alpha_weights.json",
            hysteresis_days=self.cfg.regime.hysteresis_days,
        )
        self._regime_warmed_up = False

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

        # Strategy layer
        entry_filter = EntryFilter(
            max_positions=self.cfg.trading.max_positions,
            max_trades_per_month=self.cfg.trading.max_trades_per_month,
        )
        sizer = VolTargetSizer(target_annual_vol=self.cfg.trading.target_annual_vol)
        scorer = OpportunityScorer(gate_multiplier=1.75)
        self.signal_gen = SignalGenerator(
            opportunity_scorer=scorer,
            entry_filter=entry_filter,
            sizer=sizer,
            directional_sources=DEFAULT_DIRECTIONAL,
            conviction_sources=DEFAULT_CONVICTION,
            max_positions=self.cfg.trading.max_positions,
            max_single_weight=self.cfg.trading.max_single_weight,
        )
        logger.info("Regime engine: cross-asset (Phase 2 unified)")

        # Execution
        self.executor = TradingExecutor(self.cfg)

    # ──────────────────────────────────────────────────────────
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
        return self.feature_eng.compute_all(df)

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """Generate trading signal via Phase 2 single-path flow."""
        logger.info("Generating signal...")

        # Vol predictions (for conviction)
        vol_scores = self.inference.predict(df)

        # Regime via macro composite (S3)
        regime = self._detect_regime(df)
        logger.info(regime.describe())

        # Operational state
        today = datetime.now().strftime("%Y-%m-%d")
        state = OperationalState(
            current_positions=self.executor.positions.count(),
            monthly_trades=self.executor.positions.monthly_trade_count(today),
            circuit_breaker_active=False,
        )

        # Ticker volume map (liquidity check)
        ticker_volume_map = self._build_volume_map(df)

        # Signal generation
        signal = self.signal_gen.generate(
            ohlcv=df,
            vol_scores=vol_scores,
            regime=regime,
            cost=self.cfg.trading.costs.us_roundtrip,
            state=state,
            ticker_volume_map=ticker_volume_map,
        )

        logger.info(
            f"Signal: {signal.action} — {len(signal.positions)} positions, "
            f"cash={signal.cash_weight:.0%}, rejections={signal.rejection_reasons}"
        )

        return {
            "action": signal.action,
            "positions": signal.positions,
            "regime": signal.regime_name,
            "regime_score": signal.regime_score,
            "position_scale": signal.position_scale,
            "cash_weight": signal.cash_weight,
        }

    def execute(self, signal: dict) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.executor.execute_entry(signal, today)

    def monitor(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.executor.monitor_positions(today)

    def run_session(self) -> dict:
        logger.info("=" * 60)
        logger.info(f"V3 Trading Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        df = self.collect_data()
        signal = self.generate_signal(df)
        entries = self.execute(signal)
        logger.info(f"Entries: {len(entries)}")
        exits = self.monitor()
        logger.info(f"Exits: {len(exits)}")

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

    # ── internal helpers ──────────────────────────────────────
    def _detect_regime(self, ohlcv: pd.DataFrame):
        """Single path: macro percentile → RegimeDetectorV2."""
        existing = self.macro_collector.load()
        if existing is None or len(existing) < 60:
            macro = self.macro_collector.collect()
        else:
            macro = self.macro_collector.collect(incremental_days=90)

        feats = self.macro_fe.compute(macro, ohlcv)
        pctl = self.macro_fe.compute_percentiles(feats)

        if not self._regime_warmed_up and len(pctl) >= 5:
            self.regime_detector.warm_up(pctl, days=5)
            self._regime_warmed_up = True

        return self.regime_detector.detect(pctl)

    @staticmethod
    def _build_volume_map(ohlcv: pd.DataFrame) -> dict[str, float]:
        """5-day average daily trading value per ticker (for liquidity check)."""
        if ohlcv.empty or "ticker" not in ohlcv.columns:
            return {}
        recent = ohlcv.sort_values("date").groupby("ticker").tail(5)
        agg = recent.groupby("ticker").apply(
            lambda g: float((g["close"] * g["volume"]).mean())
        )
        return agg.to_dict()
