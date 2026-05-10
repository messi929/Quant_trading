"""E2E live trading pipeline (Phase 2 S4 integration).

Single canonical path:
  OHLCV → features → VolTransformer inference → alpha_sources
  Macro → features → percentiles → RegimeDetectorV2 → Regime snapshot
  (alphas, conviction, regime, cost, state) → SignalGenerator → TradeSignal → execute
"""

from __future__ import annotations

import json
from datetime import date, datetime
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
from v3.execution.execution_quality import ExecutionQualityMonitor
from v3.strategy.allocation import AllocationEngine
from v3.strategy.alpha_sources import DEFAULT_CONVICTION, DEFAULT_DIRECTIONAL
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.diagnostics import NoTradeReasonLogger, TransferCoefficientMonitor
from v3.strategy.edge_calibrator import CalibrationConfig, EdgeCalibrator
from v3.strategy.edge_engine import EdgeEngine
from v3.strategy.edge_tier import EdgeTierSystem, TierThresholds
from v3.strategy.exit_thesis import ExitPolicy, ExitThesisEngine
from v3.strategy.feature_tracker import record_features_on_startup
from v3.strategy.opportunity import OpportunityScorer
from v3.strategy.partial_exit import PartialExitEngine, PartialExitPolicy
from v3.strategy.pyramid import PyramidPolicy, PyramidPolicyEngine
from v3.strategy.regime_v2 import RegimeDetectorV2
from v3.strategy.rotation import CapitalRotationEngine, RotationPolicy
from v3.strategy.signal import SignalGenerator
from v3.strategy.signal_decay import SignalDecayEngine, SignalDecayPolicy
from v3.strategy.sizing import VolTargetSizer
from v3.strategy.types import PositionState
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

        # V3.3 dependencies — graceful init (calibration table missing → Edge layer None)
        models_dir = Path(self.cfg.paths.models)

        # Diagnostics
        no_trade_logger = NoTradeReasonLogger(
            log_dir=models_dir / "no_trade_logs",
        )
        tc_monitor = TransferCoefficientMonitor(
            log_path=models_dir / "tc_history.jsonl",
        )
        execution_quality = ExecutionQualityMonitor(
            log_path=models_dir / "execution_quality.jsonl",
        )

        # Edge layer (calibration_table.json 있을 때만 활성)
        edge_calibrator = None
        edge_engine = None
        edge_tier = None
        calibration_path = Path("v3/config/edge_calibration.json")
        if calibration_path.exists():
            try:
                edge_calibrator = EdgeCalibrator.from_file(
                    calibration_path,
                    config=CalibrationConfig(insufficient_action="global"),
                )
                edge_engine = EdgeEngine()
                # tier_thresholds: 별도 파일 또는 기본값
                tier_thresholds_path = Path("v3/config/edge_thresholds.json")
                if tier_thresholds_path.exists():
                    edge_tier = EdgeTierSystem.from_file(tier_thresholds_path)
                else:
                    edge_tier = EdgeTierSystem(thresholds=TierThresholds())
                logger.info("V3.3 Edge layer loaded from calibration_table.json")
            except Exception as exc:
                logger.warning(
                    f"V3.3 Edge layer init failed: {exc!r} — features.edge_* will be no-op"
                )
                edge_calibrator = edge_engine = edge_tier = None
        else:
            logger.warning(
                "calibration_table.json not found — V3.3 Edge layer features inactive. "
                "Run v3/scripts/run_calibration_pipeline.py to enable."
            )
            # Issue 3 fix — escalate when downstream features depend on Edge
            f = self.cfg.features
            edge_dependent_active = bool(
                f.pyramid or f.rotation or f.allocation or f.edge_engine
            )
            if edge_dependent_active:
                logger.warning(
                    "⚠️ FEATURE GAP — calibration_table.json missing but "
                    "features.{pyramid|rotation|allocation|edge_engine} are ON. "
                    "These engines will silently no-op until calibration is "
                    "published. Build via v3/scripts/run_calibration_pipeline.py."
                )

        # Phase 3 (exit policies) — calibration 없어도 동작
        # Critical 1 fix — signal_refresh_fn lambda binds to self at call time.
        # _last_ctx 가 generate_signal 후 채워지므로 monitor 시점엔 사용 가능.
        exit_thesis = ExitThesisEngine(
            policy=ExitPolicy(
                hold_min_residual_edge=0.0,
                max_signal_staleness_hours=16.0,  # V3.3 stale fix
                refresh_at_session_start=True,
            ),
            signal_refresh_fn=lambda ticker: self._refresh_edge_candidate(ticker),
        )
        partial_exit = PartialExitEngine(policy=PartialExitPolicy())
        signal_decay = SignalDecayEngine(policy=SignalDecayPolicy())

        # Phase 4 (capital expansion)
        allocation = AllocationEngine()
        pyramid = PyramidPolicyEngine(policy=PyramidPolicy())
        rotation = CapitalRotationEngine(policy=RotationPolicy())

        # BookOptimizer with all V3.3 deps wired
        self.book_optimizer = BookOptimizer(
            signal_gen=self.signal_gen,
            features=self.cfg.features,
            edge_calibrator=edge_calibrator,
            edge_engine=edge_engine,
            edge_tier=edge_tier,
            exit_thesis=exit_thesis,
            partial_exit=partial_exit,
            signal_decay=signal_decay,
            allocation=allocation,
            pyramid=pyramid,
            rotation=rotation,
            no_trade_logger=no_trade_logger,
            tc_monitor=tc_monitor,
            execution_quality=execution_quality,
        )

        # F2 fix — record feature activations for rollback safety net
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
        self._last_ctx = None  # V3.3 DecisionContext (when features.* ON)
        # V3.3 pyramid/rotation tracking (cross-session, persisted)
        self._v33_state_path = (
            Path(self.cfg.paths.models) / "v33_session_state.json"
        )
        self._adds_per_position: dict[str, int] = {}
        self._initial_weights: dict[str, float] = {}
        self._rotations_this_month: int = 0
        self._rotations_month_key: str = ""
        self._load_v33_state()

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

        # V3.3 inputs (only relevant when features.* ON; harmless overhead OFF)
        # Pass last ctx candidates so position residual_edge metadata is accurate
        last_candidates_map = (
            {c.ticker: c for c in self._last_ctx.candidates}
            if self._last_ctx is not None else {}
        )
        position_states = self._convert_to_position_states(
            self.executor.positions.positions, df,
            candidates_by_ticker=last_candidates_map,
        )
        exit_triggers = self._check_triggers_v33(
            self.executor.positions.positions, df,
        )
        signal_age_h = self._signal_age_hours()
        rotations_this_month = self._refresh_rotations_counter(today)

        # V3.3 BookOptimizer routing (features.* OFF → V3.2.1 parity)
        ctx = self.book_optimizer.decide_with_context(
            ohlcv=df,
            vol_scores=vol_scores,
            regime=regime,
            cost=self.cfg.trading.costs.us_roundtrip,
            state=state,
            as_of=pd.Timestamp(today),
            ticker_volume_map=ticker_volume_map,
            positions=position_states,
            exit_triggers=exit_triggers,
            signal_age_hours=signal_age_h,
            rotations_this_month=rotations_this_month,
            adds_per_position=dict(self._adds_per_position),
            initial_weights=dict(self._initial_weights),
        )
        signal = ctx.signal  # parity: 기존 signal 처리 코드와 호환
        self._last_ctx = ctx

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
        if self._use_v33_routing() and self._last_ctx is not None:
            r = self.executor.execute_actions(self._last_ctx.actions, today)
            mutated = False
            for entry in r["entered"]:
                self._initial_weights.setdefault(entry["ticker"], entry["weight"])
                mutated = True
            for ro in r["rotated"]:
                self._initial_weights[ro["ticker_in"]] = ro["weight"]
                self._initial_weights.pop(ro["ticker_out"], None)
                self._adds_per_position.pop(ro["ticker_out"], None)
                self._rotations_this_month += 1
                mutated = True
            for add in r["added"]:
                self._adds_per_position[add["ticker"]] = (
                    self._adds_per_position.get(add["ticker"], 0) + 1
                )
                mutated = True
            for ex in r["exited"]:
                self._initial_weights.pop(ex["ticker"], None)
                self._adds_per_position.pop(ex["ticker"], None)
                mutated = True
            if mutated:
                self._save_v33_state()  # Critical 2 fix
            return r["entered"] + r["rotated"] + r["added"]
        return self.executor.execute_entry(signal, today)

    # ──────────────────────────────────────────────────────────
    # V3.3 helpers
    # ──────────────────────────────────────────────────────────
    def _use_v33_routing(self) -> bool:
        """V3.3 ctx.actions consumption active when any decision-changing flag ON.

        Diagnostic-only flags (no_trade_logger/tc_monitor/execution_quality)
        do NOT trigger V3.3 routing — they record passively under V3.2.1 path.
        """
        f = self.cfg.features
        return bool(
            f.exit_thesis or f.partial_exit or f.signal_decay
            or f.allocation or f.pyramid or f.rotation
        )

    def _convert_to_position_states(
        self,
        positions: list[dict],
        ohlcv: pd.DataFrame,
        candidates_by_ticker: dict | None = None,
    ) -> list[PositionState]:
        """V3.2.1 position dicts → V3.3 PositionState (uses latest close per ticker).

        Issue 4 fix — populate residual_edge from candidates_by_ticker[ticker]
        .net_edge_5d when available. Falls back to 0.0 if Edge layer inactive
        or ticker not in candidates.
        """
        if not positions:
            return []
        # Build latest-row map per ticker
        if ohlcv is None or ohlcv.empty:
            latest = {}
        else:
            sorted_df = ohlcv.sort_values("date") if "date" in ohlcv.columns else ohlcv
            latest = (
                sorted_df.groupby("ticker").tail(1).set_index("ticker").to_dict("index")
            )
        candidates_by_ticker = candidates_by_ticker or {}
        states: list[PositionState] = []
        for pos in positions:
            ticker = pos["ticker"]
            row = latest.get(ticker)
            entry_price = float(pos.get("entry_price", 0) or 0)
            current_price = (
                float(row["close"]) if row is not None and row.get("close") else entry_price
            )
            pnl = (current_price / entry_price - 1.0) if entry_price > 0 else 0.0
            cand = candidates_by_ticker.get(ticker)
            residual = (
                float(cand.net_edge_5d) if cand is not None
                and getattr(cand, "net_edge_5d", None) is not None
                else 0.0
            )
            tier = (
                getattr(cand, "edge_tier", None) or "A"
            ) if cand is not None else "A"
            states.append(PositionState(
                ticker=ticker,
                entry_date=pd.Timestamp(str(pos.get("entry_date", ""))[:10] or "1970-01-01"),
                entry_price=entry_price,
                current_price=current_price,
                current_weight=float(pos.get("weight", 0)),
                entry_edge=float(pos.get("opportunity", 0.0)),
                residual_edge=residual,
                entry_tier="A",
                current_tier=tier,
                pnl_pct=pnl,
                max_unrealized_pnl_pct=max(pnl, 0.0),
                drawdown_from_peak_pct=min(pnl, 0.0),
                holding_days=int(pos.get("hold_days", 0)) + 1,
                dominant_alpha="trend",
                expected_holding_days=10,
                thesis_alive=pnl > 0,
            ))
        return states

    def _check_triggers_v33(
        self, positions: list[dict], ohlcv: pd.DataFrame,
    ) -> dict[str, str]:
        """ExitRules → {ticker: trigger_name} for BookOptimizer exit_triggers."""
        if not positions or ohlcv is None or ohlcv.empty:
            return {}
        sorted_df = ohlcv.sort_values("date") if "date" in ohlcv.columns else ohlcv
        latest = (
            sorted_df.groupby("ticker").tail(1).set_index("ticker").to_dict("index")
        )
        triggers: dict[str, str] = {}
        for pos in positions:
            row = latest.get(pos["ticker"])
            if row is None:
                continue
            current_price = float(row.get("close", 0) or 0)
            current_vol = float(row.get("vol_cc_5d", pos.get("entry_vol", 0.3)) or 0.3)
            hold_days = int(pos.get("hold_days", 0)) + 1
            decision = self.executor.exit_rules.check(
                entry_price=float(pos["entry_price"]),
                current_price=current_price,
                hold_days=hold_days,
                entry_vol=float(pos.get("entry_vol", 0.3)),
                current_vol=current_vol,
                confidence=float(pos.get("confidence", 0.5)),
                low_price=float(row.get("low", current_price)),
            )
            triggers[pos["ticker"]] = (
                decision.reason if decision.should_exit else "max_hold"
            )
        return triggers

    def _signal_age_hours(self) -> float:
        """Hours since last opportunity snapshot (0 if first run)."""
        if self._opportunity_at is None:
            return 0.0
        return (datetime.now() - self._opportunity_at).total_seconds() / 3600.0

    def _refresh_rotations_counter(self, today: str) -> int:
        """Reset rotations_this_month counter on calendar-month boundary."""
        month_key = today[:7]
        if month_key != self._rotations_month_key:
            self._rotations_month_key = month_key
            self._rotations_this_month = 0
            self._save_v33_state()
        return self._rotations_this_month

    # ──────────────────────────────────────────────────────────
    # V3.3 cross-session state persistence (Critical 2 fix)
    # ──────────────────────────────────────────────────────────
    def _load_v33_state(self) -> None:
        """Load pyramid/rotation tracking state from JSON (if exists)."""
        if not self._v33_state_path.exists():
            return
        try:
            data = json.loads(self._v33_state_path.read_text(encoding="utf-8"))
            self._adds_per_position = dict(data.get("adds_per_position", {}))
            self._initial_weights = dict(data.get("initial_weights", {}))
            self._rotations_this_month = int(data.get("rotations_this_month", 0))
            self._rotations_month_key = str(data.get("rotations_month_key", ""))
            logger.info(
                f"V3.3 session state loaded: {len(self._adds_per_position)} adds, "
                f"{len(self._initial_weights)} weights, "
                f"{self._rotations_this_month} rotations in {self._rotations_month_key}"
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(f"V3.3 state load failed ({exc!r}) — starting fresh")

    def _save_v33_state(self) -> None:
        """Persist pyramid/rotation tracking state."""
        try:
            self._v33_state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "adds_per_position": self._adds_per_position,
                "initial_weights": self._initial_weights,
                "rotations_this_month": self._rotations_this_month,
                "rotations_month_key": self._rotations_month_key,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._v33_state_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error(f"V3.3 state save failed: {exc!r}")

    def _refresh_edge_candidate(self, ticker: str):
        """Critical 1 fix — ExitThesisEngine.signal_refresh_fn callback.

        Returns latest EdgeCandidate for ticker from current ctx, or None.
        ExitThesis treats None as 'no fresh signal' → conservative EXIT.
        Lambda in __init__ binds to self at call time; safe even though
        _last_ctx is initialized after this method is referenced.
        """
        if self._last_ctx is None:
            return None
        for c in self._last_ctx.candidates:
            if c.ticker == ticker:
                return c
        return None

    def monitor(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        # V3.3 path: features.exit_thesis ON + DecisionContext available.
        # ExitThesisEngine uses its own staleness threshold (16h) to decide
        # whether to refresh — V3.2.1 8h cache drop is unnecessary here.
        if (
            self.cfg.features.exit_thesis
            and self._last_ctx is not None
            and self.book_optimizer.exit_thesis is not None
        ):
            return self._monitor_v33(today)
        # V3.2.1 path (legacy ExitRules + inline conditional veto).
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

    def _monitor_v33(self, today: str) -> list[dict]:
        """V3.3 monitor — ExitThesis re-evaluation per held position.

        For each held position:
          1. Fetch real-time price via executor._get_price (paper or KIS).
          2. Compute ExitRules trigger (same as V3.2.1).
          3. If trigger fires, build PositionState + call ExitThesisEngine.decide().
             Otherwise skip (KEEP).
          4. Aggregate emitted BookActions and route through execute_actions().

        Replacement candidates come from `self._last_ctx.candidates` (last
        session). signal_age_hours uses _opportunity_at — ExitThesisEngine's
        16h staleness threshold + signal_refresh_fn callback decides what to do.
        """
        from v3.strategy.types import BookAction  # local import (deferred)

        positions = list(self.executor.positions.positions)
        if not positions:
            return []

        ctx = self._last_ctx
        candidates_by_ticker = {c.ticker: c for c in ctx.candidates}
        held = {p["ticker"] for p in positions}
        replacement_pool = [
            c for c in ctx.candidates
            if c.ticker not in held and c.net_edge_5d is not None
        ]
        signal_age_h = self._signal_age_hours()
        engine = self.book_optimizer.exit_thesis
        actions: list[BookAction] = []

        for pos in positions:
            ticker = pos["ticker"]
            price_info = self.executor._get_price(ticker)
            if not price_info:
                continue
            current_price = price_info.get("price", 0)
            if current_price <= 0:
                continue
            current_vol = price_info.get("vol", pos.get("entry_vol", 0.2))
            low_price = price_info.get("low", current_price)
            entry_date_str = str(pos.get("entry_date", today))[:10]
            try:
                entry_d = date.fromisoformat(entry_date_str)
                today_d = date.fromisoformat(today)
                hold_days = max(0, (today_d - entry_d).days)
            except ValueError:
                hold_days = int(pos.get("hold_days", 0) or 0)

            decision = self.executor.exit_rules.check(
                entry_price=pos["entry_price"],
                current_price=current_price,
                hold_days=hold_days,
                entry_vol=pos.get("entry_vol", 0.2),
                current_vol=current_vol,
                confidence=pos.get("confidence", 0.5),
                low_price=low_price,
            )
            if not decision.should_exit:
                continue

            current_edge = candidates_by_ticker.get(ticker)
            state = self._convert_one_position_state(
                pos, current_price, current_edge=current_edge,
            )
            action = engine.decide(
                position=state,
                exit_trigger=decision.reason,
                current_edge=current_edge,
                replacement_candidates=replacement_pool,
                signal_age_hours=signal_age_h,
            )
            actions.append(action)
            logger.info(
                f"V3.3 monitor {ticker}: trigger={decision.reason} "
                f"→ {action.action_type} ({action.reason})"
            )

        if not actions:
            return []
        result = self.executor.execute_actions(actions, today)
        # Update cross-session counters for any rotations triggered here
        mutated = False
        for ro in result.get("rotated", []):
            self._initial_weights[ro["ticker_in"]] = ro["weight"]
            self._initial_weights.pop(ro["ticker_out"], None)
            self._adds_per_position.pop(ro["ticker_out"], None)
            self._rotations_this_month += 1
            mutated = True
        for ex in result.get("exited", []):
            self._initial_weights.pop(ex["ticker"], None)
            self._adds_per_position.pop(ex["ticker"], None)
            mutated = True
        if mutated:
            self._save_v33_state()  # Critical 2 fix
        return (
            result.get("exited", [])
            + result.get("trimmed", [])
            + result.get("rotated", [])
        )

    def _convert_one_position_state(
        self, pos: dict, current_price: float,
        current_edge=None,
    ):
        """Build a single PositionState from V3.2.1 dict + live current_price.

        Issue 4 fix — accept current_edge to populate residual_edge from
        net_edge_5d when Edge layer active.
        """
        from v3.strategy.types import PositionState
        entry_price = float(pos.get("entry_price", 0) or 0)
        pnl = (current_price / entry_price - 1.0) if entry_price > 0 else 0.0
        residual = (
            float(current_edge.net_edge_5d) if current_edge is not None
            and getattr(current_edge, "net_edge_5d", None) is not None
            else 0.0
        )
        tier = (
            getattr(current_edge, "edge_tier", None) or "A"
        ) if current_edge is not None else "A"
        return PositionState(
            ticker=pos["ticker"],
            entry_date=pd.Timestamp(str(pos.get("entry_date", ""))[:10] or "1970-01-01"),
            entry_price=entry_price,
            current_price=current_price,
            current_weight=float(pos.get("weight", 0)),
            entry_edge=float(pos.get("opportunity", 0.0)),
            residual_edge=residual,
            entry_tier="A",
            current_tier=tier,
            pnl_pct=pnl,
            max_unrealized_pnl_pct=max(pnl, 0.0),
            drawdown_from_peak_pct=min(pnl, 0.0),
            holding_days=int(pos.get("hold_days", 0)) + 1,
            dominant_alpha="trend",
            expected_holding_days=10,
            thesis_alive=pnl > 0,
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
