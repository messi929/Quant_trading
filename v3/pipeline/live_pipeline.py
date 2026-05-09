"""E2E live trading pipeline (Phase 2 S4 integration).

Single canonical path:
  OHLCV → features → VolTransformer inference → alpha_sources
  Macro → features → percentiles → RegimeDetectorV2 → Regime snapshot
  (alphas, conviction, regime, cost, state) → SignalGenerator → TradeSignal → execute
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

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
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.feature_tracker import record_features_on_startup
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

        # V3.3 BookOptimizer (PR-2.5 integration).
        # features.* OFF default → V3.2.1 100% parity.
        # Phase 2/3/4 dependencies wired conditionally as features activate.
        self.book_optimizer = BookOptimizer(
            signal_gen=self.signal_gen,
            features=self.cfg.features,
        )

        # F2 fix — record feature activations for rollback safety net
        models_dir = Path(self.cfg.paths.models)
        record_features_on_startup(
            current=self.cfg.features,
            history_path=models_dir / "feature_activations.jsonl",
            snapshot_path=models_dir / "feature_state_snapshot.json",
            activated_by="live_pipeline_startup",
        )

        # Execution
        self.executor = TradingExecutor(self.cfg)

        # Latest opportunity snapshot (for TP conditional re-evaluation)
        self._opportunity_map: dict[str, float] = {}
        self._opportunity_gate: float = 0.0
        self._opportunity_at: datetime | None = None  # when the snapshot was taken
        self._last_signal = None

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

        # V3.3 BookOptimizer routing (features.* OFF → V3.2.1 parity)
        ctx = self.book_optimizer.decide_with_context(
            ohlcv=df,
            vol_scores=vol_scores,
            regime=regime,
            cost=self.cfg.trading.costs.us_roundtrip,
            state=state,
            as_of=pd.Timestamp(today),
            ticker_volume_map=ticker_volume_map,
        )
        signal = ctx.signal  # parity: 기존 signal 처리 코드와 호환

        logger.info(
            f"Signal: {signal.action} — {len(signal.positions)} positions, "
            f"cash={signal.cash_weight:.0%}, rejections={signal.rejection_reasons}"
        )

        # Cache opportunity snapshot for TP conditional re-evaluation during monitor
        self._opportunity_map = dict(signal.opportunity_map)
        self._opportunity_gate = signal.opportunity_gate
        self._opportunity_at = datetime.now()
        self._last_signal = signal

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
        # Stale-cache safeguard: if opportunity snapshot is >8h old
        # (spans a session gap), drop it to avoid vetoing with outdated alpha.
        if self._opportunity_at is not None:
            age_h = (datetime.now() - self._opportunity_at).total_seconds() / 3600
            if age_h > 8:
                logger.warning(
                    f"Opportunity cache stale ({age_h:.1f}h old) — "
                    f"dropping for TP veto (safer to let TP fire unconditionally)"
                )
                opp_map, opp_gate = {}, 0.0
            else:
                opp_map, opp_gate = self._opportunity_map, self._opportunity_gate
        else:
            opp_map, opp_gate = {}, 0.0
        return self.executor.monitor_positions(
            today,
            opportunity_map=opp_map,
            opportunity_gate=opp_gate,
        )

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
        self._log_recommendation_snapshot(summary, entries, exits)
        return summary

    def _log_recommendation_snapshot(
        self, summary: dict, entries: list[dict], exits: list[dict]
    ) -> None:
        """Append per-session recommendation snapshot to JSONL log.

        Observation tool only — does not affect policy. Captures top-10
        opportunities, sized positions, rejections, and execution outcome
        so we can distinguish 'market quiet' from 'sizer too conservative'.
        """
        sig = getattr(self, "_last_signal", None)
        if sig is None:
            return
        sorted_opps = sorted(sig.opportunity_map.items(), key=lambda x: -x[1])[:10]
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "date": summary["date"],
            "regime": {
                "name": sig.regime_name,
                "score": round(sig.regime_score, 4),
                "scale": round(sig.position_scale, 4),
            },
            "n_candidates": sig.n_candidates,
            "n_approved": sig.n_approved,
            "opp_gate": round(sig.opportunity_gate, 6),
            "top_opportunities": [
                {"ticker": t, "opp": round(o, 6)} for t, o in sorted_opps
            ],
            "selected_positions": [
                {
                    "ticker": p["ticker"],
                    "weight": round(p.get("weight", 0.0), 4),
                    "opportunity": round(p.get("opportunity", 0.0), 6),
                    "conviction": round(p.get("conviction", 0.0), 4),
                    "direction": round(p.get("direction", 0.0), 6),
                }
                for p in sig.positions
            ],
            "rejections": dict(sig.rejection_reasons),
            "entries_count": len(entries),
            "entered_tickers": [e.get("ticker", "?") for e in entries],
            "exits_count": len(exits),
            "exited": [
                {"ticker": e.get("ticker", "?"), "reason": e.get("reason", "?")}
                for e in exits
            ],
            "open_positions": summary["open_positions"],
        }
        log_path = Path(self.cfg.paths.models) / "recommendation_log.jsonl"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                # default=float covers numpy.float32/64 from inference;
                # broad exception below catches anything else (e.g., custom types).
                f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to write recommendation log: {exc}")

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
