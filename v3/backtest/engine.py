"""V3 Backtest Engine — Phase 2 rewrite.

Uses the SAME SignalGenerator as live_pipeline (Phase 2 principle #6:
live-backtest code path parity). The engine only handles:
  - Date iteration
  - Per-date slicing of OHLCV up to t (no look-ahead)
  - Entry execution at t+1 open with slippage
  - Exit via ExitRules (unchanged from Phase 1)
  - PnL tracking, equity curve, metrics

Legacy Phase 1 engine preserved at v3/backtest/_legacy/engine_phase1.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from v3.config.schema import V3Config
from v3.data.macro_collector import MacroCollector
from v3.data.macro_features import MacroFeatureEngineer
from v3.data.feature_engineer import VolFeatureEngineer
from v3.rules.entry import EntryFilter, OperationalState
from v3.rules.exit import ExitRules
from v3.execution.execution_quality import ExecutionQualityMonitor
from v3.strategy.allocation import AllocationEngine
from v3.strategy.alpha_sources import DEFAULT_CONVICTION, DEFAULT_DIRECTIONAL
from v3.strategy.book_optimizer import BookOptimizer
from v3.strategy.diagnostics import NoTradeReasonLogger, TransferCoefficientMonitor
from v3.strategy.edge_calibrator import CalibrationConfig, EdgeCalibrator
from v3.strategy.edge_engine import EdgeEngine
from v3.strategy.edge_tier import EdgeTierSystem, TierThresholds
from v3.strategy.exit_thesis import ExitPolicy, ExitThesisEngine
from v3.strategy.opportunity import OpportunityScorer
from v3.strategy.partial_exit import PartialExitEngine, PartialExitPolicy
from v3.strategy.pyramid import PyramidPolicy, PyramidPolicyEngine
from v3.strategy.regime_v2 import RegimeDetectorV2
from v3.strategy.risk import RiskManager
from v3.strategy.rotation import CapitalRotationEngine, RotationPolicy
from v3.strategy.signal import SignalGenerator
from v3.strategy.signal_decay import SignalDecayEngine, SignalDecayPolicy
from v3.strategy.sizing import VolTargetSizer
from v3.strategy.types import BookAction, DecisionContext, PositionState


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    ticker: str
    market: str
    direction: str
    entry_price: float
    exit_price: float
    weight: float
    gross_return: float
    net_return: float
    cost: float
    exit_reason: str
    hold_days: int
    vol_score: float
    opportunity: float


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    avg_hold_days: float
    total_trades: int
    avg_trades_per_month: float
    total_cost: float
    profit_factor: float
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame


class BacktestEngine:
    """Event-driven backtest engine using Phase 2 SignalGenerator."""

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.feature_eng = VolFeatureEngineer()

        self.macro_collector = MacroCollector(
            save_dir=cfg.paths.raw_data,
            history_years=cfg.regime.macro_history_years,
        )
        self.macro_fe = MacroFeatureEngineer()

        self.regime_detector = RegimeDetectorV2(
            weights_path="v3/config/alpha_weights.json",
            hysteresis_days=cfg.regime.hysteresis_days,
        )

        self.exit_rules = ExitRules(
            profit_take_pct=cfg.trading.profit_take_pct,
            max_hold_days=cfg.trading.max_hold_days,
            vol_contraction_ratio=cfg.trading.vol_contraction_exit_ratio,
            daily_loss_limit=cfg.risk.daily_loss_limit,
        )
        self.risk_manager = RiskManager(
            daily_loss_limit=cfg.risk.daily_loss_limit,
            total_loss_limit=cfg.risk.total_loss_limit,
        )

        # Signal generator (live-parity)
        entry_filter = EntryFilter(
            max_positions=cfg.trading.max_positions,
            max_trades_per_month=cfg.trading.max_trades_per_month,
        )
        sizer = VolTargetSizer(target_annual_vol=cfg.trading.target_annual_vol)
        scorer = OpportunityScorer(gate_multiplier=1.75)
        self.signal_gen = SignalGenerator(
            opportunity_scorer=scorer,
            entry_filter=entry_filter,
            sizer=sizer,
            directional_sources=DEFAULT_DIRECTIONAL,
            conviction_sources=DEFAULT_CONVICTION,
            max_positions=cfg.trading.max_positions,
            max_single_weight=cfg.trading.max_single_weight,
        )

        # V3.3 dependencies — graceful init (calibration table missing → Edge layer None)
        from pathlib import Path

        # Diagnostics
        models_dir = Path(cfg.paths.models)
        no_trade_logger = NoTradeReasonLogger(
            log_dir=models_dir / "no_trade_logs",
        )
        tc_monitor = TransferCoefficientMonitor(
            log_path=models_dir / "tc_history.jsonl",
        )
        execution_quality = ExecutionQualityMonitor(
            log_path=models_dir / "execution_quality.jsonl",
        )

        # Edge layer (optional — calibration table 있을 때만)
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
                tier_thresholds_path = Path("v3/config/edge_thresholds.json")
                if tier_thresholds_path.exists():
                    edge_tier = EdgeTierSystem.from_file(tier_thresholds_path)
                else:
                    edge_tier = EdgeTierSystem(thresholds=TierThresholds())
            except Exception as exc:
                logger.warning(
                    f"Backtest Edge layer init failed: {exc!r} — features.edge_* no-op"
                )
                edge_calibrator = edge_engine = edge_tier = None

        # Phase 3 + 4 (always available)
        exit_thesis = ExitThesisEngine(
            policy=ExitPolicy(
                hold_min_residual_edge=0.0,
                max_signal_staleness_hours=16.0,
            ),
        )
        partial_exit = PartialExitEngine(policy=PartialExitPolicy())
        signal_decay = SignalDecayEngine(policy=SignalDecayPolicy())
        allocation = AllocationEngine()
        pyramid = PyramidPolicyEngine(policy=PyramidPolicy())
        rotation = CapitalRotationEngine(policy=RotationPolicy())

        self.book_optimizer = BookOptimizer(
            signal_gen=self.signal_gen,
            features=cfg.features,
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

    def run(
        self,
        df: pd.DataFrame,
        vol_predictions: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """Run backtest simulation.

        Thin wrapper around _run_impl that guarantees flush_diagnostics()
        runs even if simulation raises. test_regression Bug 12 invariant.
        """
        try:
            return self._run_impl(df, vol_predictions, initial_capital)
        finally:
            self.book_optimizer.flush_diagnostics()

    def _run_impl(
        self,
        df: pd.DataFrame,
        vol_predictions: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """Backtest simulation core (no flush — done by caller's finally)."""
        capital = initial_capital or self.cfg.backtest.initial_capital
        portfolio_value = capital
        cash = capital
        positions: list[dict] = []
        trades: list[TradeRecord] = []
        equity_records: list[dict] = []

        # Precompute features and macro percentiles once
        df_feat = self.feature_eng.compute_all(df) if "vol_cc_5d" not in df.columns else df
        macro = self.macro_collector.load()
        if macro is None or len(macro) < 60:
            logger.warning("Macro data missing; detector will fallback to uniform")
            macro_pctl = pd.DataFrame()
        else:
            macro_feats = self.macro_fe.compute(macro, df_feat)
            macro_pctl = self.macro_fe.compute_percentiles(macro_feats)

        # Warm up hysteresis on first half of macro history
        if not macro_pctl.empty:
            self.regime_detector.warm_up(macro_pctl, days=10)

        pred_dates = sorted(vol_predictions["date"].unique())
        logger.info(
            f"Backtest: {len(pred_dates)} days, capital={capital:,.0f}, "
            f"range: {pred_dates[0]} ~ {pred_dates[-1]}"
        )

        current_month: str | None = None
        monthly_unique_tickers: set[str] = set()  # Phase 25.1 옵션 C

        for date in pred_dates:
            date_ts = pd.Timestamp(date)
            date_str = str(date)[:10]
            month = date_str[:7]
            if month != current_month:
                current_month = month
                monthly_unique_tickers = set()

            today_data = df_feat[df_feat["date"] == date_ts]
            if today_data.empty:
                continue

            self.risk_manager.update(portfolio_value)

            circuit_breaker_active = self.risk_manager.circuit_breaker_scale(portfolio_value) < 1.0
            df_up_to_t = df_feat[df_feat["date"] <= date_ts]
            vol_scores_today = vol_predictions[vol_predictions["date"] == date][
                ["ticker", "vol_score", "confidence"]
            ]
            ticker_volume_map = self._build_volume_map(today_data)

            # F3 fix: features.exit_thesis ON 시 V3.3 ctx.actions path 사용
            if (
                self.cfg.features.exit_thesis
                and not vol_scores_today.empty
                and not circuit_breaker_active
            ):
                regime = (
                    self.regime_detector.detect(macro_pctl, as_of=date_ts)
                    if not macro_pctl.empty else None
                )
                if regime is not None:
                    # V3.3 path
                    deployed_pre = 1.0 - (cash / max(portfolio_value, 1))
                    triggers = self._check_triggers_v33(
                        positions, today_data, deployed_pre,
                    )
                    position_states = self._convert_to_position_states(
                        positions, today_data,
                    )
                    state = OperationalState(
                        current_positions=len(positions),
                        monthly_trades=len(monthly_unique_tickers),
                        circuit_breaker_active=circuit_breaker_active,
                    )
                    ctx = self.book_optimizer.decide_with_context(
                        ohlcv=df_up_to_t,
                        vol_scores=vol_scores_today,
                        regime=regime,
                        cost=self._cost_for_market(today_data),
                        state=state,
                        as_of=date_ts,
                        ticker_volume_map=ticker_volume_map,
                        positions=position_states,
                        exit_triggers=triggers,
                    )
                    positions, new_trades, cash, daily_pnl = (
                        self._process_book_actions(
                            ctx, positions, today_data, vol_scores_today,
                            date_str, cash, monthly_unique_tickers,
                        )
                    )
                    trades.extend(new_trades)
                else:
                    # No regime → fall through to V3.2.1 _check_exits
                    positions, new_trades, cash, daily_pnl = self._check_exits(
                        positions, today_data, date_str, cash
                    )
                    trades.extend(new_trades)
            else:
                # V3.2.1 path (parity preserved when features.exit_thesis OFF)

                # 1. Exit check on existing positions
                positions, new_trades, cash, daily_pnl = self._check_exits(
                    positions, today_data, date_str, cash
                )
                trades.extend(new_trades)

                # 2. Entry via SignalGenerator (live-parity)
                state = OperationalState(
                    current_positions=len(positions),
                    monthly_trades=len(monthly_unique_tickers),
                    circuit_breaker_active=circuit_breaker_active,
                )

                if not vol_scores_today.empty and not circuit_breaker_active:
                    regime = (
                        self.regime_detector.detect(macro_pctl, as_of=date_ts)
                        if not macro_pctl.empty else None
                    )
                    if regime is None:
                        pass  # Skip entries when no regime
                    else:
                        # V3.3 BookOptimizer routing (features.* OFF → V3.2.1 parity)
                        ctx = self.book_optimizer.decide_with_context(
                            ohlcv=df_up_to_t,
                            vol_scores=vol_scores_today,
                            regime=regime,
                            cost=self._cost_for_market(today_data),
                            state=state,
                            as_of=date_ts,
                            ticker_volume_map=ticker_volume_map,
                        )
                        signal = ctx.signal  # parity: 기존 entry 처리 코드와 호환

                        if signal.action == "TRADE":
                            for pos_dict in signal.positions:
                                ticker = pos_dict["ticker"]
                                if any(p["ticker"] == ticker for p in positions):
                                    continue

                                ticker_row = today_data[today_data["ticker"] == ticker]
                                if ticker_row.empty:
                                    continue
                                row = ticker_row.iloc[0]

                                market = self._market_of(ticker)
                                ticker_vol = float(row.get("vol_cc_5d", 0.2) or 0.2)
                                slippage = self.get_realistic_slippage(market, ticker_vol)
                                entry_price = float(row["open"]) * (1 + slippage)
                                if entry_price <= 0:
                                    continue

                                weight = pos_dict["weight"]
                                allocated = cash * weight
                                if allocated < self.cfg.trading.min_order_amount_krw:
                                    continue

                                init_unrealized = (
                                    row["close"] / entry_price - 1
                                    if entry_price > 0 else 0.0
                                )
                                positions.append({
                                    "ticker": ticker,
                                    "entry_date": date_str,
                                    "entry_price": entry_price,
                                    "weight": weight,
                                    "hold_days": 0,
                                    "entry_vol": ticker_vol,
                                    "vol_score": (
                                        vol_scores_today[vol_scores_today["ticker"] == ticker]
                                        ["vol_score"].iloc[0]
                                        if ticker in vol_scores_today["ticker"].values else 0.0
                                    ),
                                    "opportunity": pos_dict.get("opportunity", 0.0),
                                    "capital_allocated": allocated,
                                    "last_unrealized": init_unrealized,
                                })
                                cash -= allocated
                                monthly_unique_tickers.add(ticker)

                                if len(positions) >= self.cfg.trading.max_positions:
                                    break

            # 3. Portfolio mark-to-market
            invested = sum(
                p["capital_allocated"] * (1 + p.get("last_unrealized", 0))
                for p in positions
            )
            portfolio_value = cash + invested

            equity_records.append({
                "date": date_str,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "n_positions": len(positions),
                "daily_pnl": daily_pnl,
            })

        # Force close remaining at end
        for pos in positions:
            last_data = df_feat[df_feat["ticker"] == pos["ticker"]].sort_values("date").iloc[-1]
            market = self._market_of(pos["ticker"])
            cost = self.cfg.cost_for_market(market)
            gross_ret = last_data["close"] / pos["entry_price"] - 1
            net_ret = gross_ret - cost
            trades.append(TradeRecord(
                entry_date=pos["entry_date"],
                exit_date=str(last_data["date"])[:10],
                ticker=pos["ticker"],
                market=market,
                direction="long",
                entry_price=pos["entry_price"],
                exit_price=last_data["close"],
                weight=pos["weight"],
                gross_return=gross_ret,
                net_return=net_ret,
                cost=cost,
                exit_reason="backtest_end",
                hold_days=pos["hold_days"],
                vol_score=pos.get("vol_score", 0),
                opportunity=pos.get("opportunity", 0.0),
            ))

        equity_df = pd.DataFrame(equity_records) if equity_records else pd.DataFrame()
        result = self._compute_metrics(trades, equity_df, capital)

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"return={result.total_return:.2%}, Sharpe={result.sharpe:.2f}, "
            f"MDD={result.max_drawdown:.2%}"
        )
        return result

    # ── internal ──────────────────────────────────────────
    def _check_exits(
        self,
        positions: list[dict],
        today_data: pd.DataFrame,
        date_str: str,
        cash: float,
    ) -> tuple[list[dict], list[TradeRecord], float, float]:
        """Run exit rules; returns (remaining_positions, new_trades, cash, daily_pnl)."""
        new_positions: list[dict] = []
        new_trades: list[TradeRecord] = []
        daily_pnl = 0.0

        for pos in positions:
            tdata = today_data[today_data["ticker"] == pos["ticker"]]
            if tdata.empty:
                new_positions.append(pos)
                continue
            row = tdata.iloc[0]
            current_price = float(row["close"])
            current_vol = float(row.get("vol_cc_5d", pos.get("entry_vol", 0.3)) or 0.3)
            hold_days = pos["hold_days"] + 1

            deployed = 1.0 - (cash / max(sum(p["capital_allocated"] for p in positions) + cash, 1))
            exit_decision = self.exit_rules.check(
                entry_price=pos["entry_price"],
                current_price=current_price,
                hold_days=hold_days,
                entry_vol=pos.get("entry_vol", 0.3),
                current_vol=current_vol,
                confidence=0.5,
                low_price=float(row.get("low", current_price)),
                portfolio_deployed=deployed,
            )

            if exit_decision.should_exit:
                market = self._market_of(pos["ticker"])
                cost = self.cfg.cost_for_market(market)
                slippage = self.get_realistic_slippage(market, current_vol)
                exit_price = current_price * (1 - slippage)
                gross_ret = exit_price / pos["entry_price"] - 1
                net_ret = gross_ret - cost

                new_trades.append(TradeRecord(
                    entry_date=pos["entry_date"],
                    exit_date=date_str,
                    ticker=pos["ticker"],
                    market=market,
                    direction="long",
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    weight=pos["weight"],
                    gross_return=gross_ret,
                    net_return=net_ret,
                    cost=cost,
                    exit_reason=exit_decision.reason,
                    hold_days=hold_days,
                    vol_score=pos.get("vol_score", 0),
                    opportunity=pos.get("opportunity", 0.0),
                ))
                daily_pnl += net_ret * pos["weight"]
                cash += pos["capital_allocated"] * (1 + net_ret)
            else:
                pos["hold_days"] = hold_days
                unrealized = current_price / pos["entry_price"] - 1
                prev_unrealized = pos.get("last_unrealized", unrealized)
                daily_pnl += (unrealized - prev_unrealized) * pos["weight"]
                pos["last_unrealized"] = unrealized
                new_positions.append(pos)

        return new_positions, new_trades, cash, daily_pnl

    # ──────────────────────────────────────────────────────────────
    # F3 — V3.3 ctx.actions consumption path (features.exit_thesis ON)
    # ──────────────────────────────────────────────────────────────
    def _convert_to_position_states(
        self,
        positions: list[dict],
        today_data: pd.DataFrame,
    ) -> list[PositionState]:
        """V3.2.1 dict positions → V3.3 PositionState for BookOptimizer."""
        states: list[PositionState] = []
        for pos in positions:
            ticker = pos["ticker"]
            tdata = today_data[today_data["ticker"] == ticker]
            current_price = (
                float(tdata.iloc[0]["close"])
                if not tdata.empty else float(pos["entry_price"])
            )
            entry_price = float(pos["entry_price"])
            pnl_pct = current_price / entry_price - 1.0 if entry_price > 0 else 0.0
            states.append(PositionState(
                ticker=ticker,
                entry_date=pd.Timestamp(pos["entry_date"]),
                entry_price=entry_price,
                current_price=current_price,
                current_weight=float(pos["weight"]),
                entry_edge=float(pos.get("opportunity", 0.0)),
                residual_edge=0.0,
                entry_tier="A",
                current_tier="A",
                pnl_pct=pnl_pct,
                max_unrealized_pnl_pct=max(pnl_pct, 0.0),
                drawdown_from_peak_pct=min(pnl_pct, 0.0),
                holding_days=int(pos.get("hold_days", 0)) + 1,
                dominant_alpha="trend",
                expected_holding_days=10,
                thesis_alive=pnl_pct > 0,
            ))
        return states

    def _check_triggers_v33(
        self,
        positions: list[dict],
        today_data: pd.DataFrame,
        portfolio_deployed: float,
    ) -> dict[str, str]:
        """ExitRules.check 결과를 ticker → trigger_name dict로."""
        triggers: dict[str, str] = {}
        for pos in positions:
            tdata = today_data[today_data["ticker"] == pos["ticker"]]
            if tdata.empty:
                continue
            row = tdata.iloc[0]
            current_price = float(row["close"])
            current_vol = float(
                row.get("vol_cc_5d", pos.get("entry_vol", 0.3)) or 0.3
            )
            hold_days = int(pos.get("hold_days", 0)) + 1
            decision = self.exit_rules.check(
                entry_price=float(pos["entry_price"]),
                current_price=current_price,
                hold_days=hold_days,
                entry_vol=float(pos.get("entry_vol", 0.3)),
                current_vol=current_vol,
                confidence=0.5,
                low_price=float(row.get("low", current_price)),
                portfolio_deployed=portfolio_deployed,
            )
            triggers[pos["ticker"]] = (
                decision.reason if decision.should_exit else "max_hold"
            )
        return triggers

    def _handle_exit_action(
        self,
        action: BookAction,
        positions: list[dict],
        today_data: pd.DataFrame,
        date_str: str,
    ) -> tuple[list[dict], TradeRecord | None, float, float]:
        """EXIT/ROTATE BookAction → trade + position 제거.

        Returns: (new_positions, trade_or_None, cash_delta, pnl_delta)
        cash_delta = capital recovered (positive)
        """
        idx = next(
            (i for i, p in enumerate(positions) if p["ticker"] == action.ticker),
            None,
        )
        if idx is None:
            return positions, None, 0.0, 0.0
        pos = positions[idx]

        tdata = today_data[today_data["ticker"] == pos["ticker"]]
        if tdata.empty:
            return positions, None, 0.0, 0.0
        row = tdata.iloc[0]
        current_price = float(row["close"])
        current_vol = float(
            row.get("vol_cc_5d", pos.get("entry_vol", 0.3)) or 0.3
        )

        market = self._market_of(pos["ticker"])
        cost = self.cfg.cost_for_market(market)
        slippage = self.get_realistic_slippage(market, current_vol)
        exit_price = current_price * (1 - slippage)
        gross_ret = exit_price / pos["entry_price"] - 1.0
        net_ret = gross_ret - cost

        trade = TradeRecord(
            entry_date=pos["entry_date"],
            exit_date=date_str,
            ticker=pos["ticker"],
            market=market,
            direction="long",
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            weight=pos["weight"],
            gross_return=gross_ret,
            net_return=net_ret,
            cost=cost,
            exit_reason=action.reason,
            hold_days=int(pos.get("hold_days", 0)) + 1,
            vol_score=pos.get("vol_score", 0),
            opportunity=pos.get("opportunity", 0.0),
        )
        cash_delta = pos["capital_allocated"] * (1 + net_ret)
        pnl_delta = net_ret * pos["weight"]
        new_positions = positions[:idx] + positions[idx + 1:]
        return new_positions, trade, cash_delta, pnl_delta

    def _handle_trim_action(
        self,
        action: BookAction,
        positions: list[dict],
        today_data: pd.DataFrame,
        date_str: str,
    ) -> tuple[list[dict], TradeRecord | None, float, float]:
        """TRIM BookAction → 부분 trade + position weight 감소."""
        idx = next(
            (i for i, p in enumerate(positions) if p["ticker"] == action.ticker),
            None,
        )
        if idx is None:
            return positions, None, 0.0, 0.0
        pos = positions[idx]
        if pos["weight"] <= 0:
            return positions, None, 0.0, 0.0
        trim_fraction = 1.0 - (action.target_weight / pos["weight"])
        if trim_fraction <= 1e-9:
            return positions, None, 0.0, 0.0

        tdata = today_data[today_data["ticker"] == pos["ticker"]]
        if tdata.empty:
            return positions, None, 0.0, 0.0
        row = tdata.iloc[0]
        current_price = float(row["close"])
        current_vol = float(
            row.get("vol_cc_5d", pos.get("entry_vol", 0.3)) or 0.3
        )

        market = self._market_of(pos["ticker"])
        cost = self.cfg.cost_for_market(market)
        slippage = self.get_realistic_slippage(market, current_vol)
        exit_price = current_price * (1 - slippage)
        gross_ret = exit_price / pos["entry_price"] - 1.0
        net_ret = gross_ret - cost

        trim_capital = pos["capital_allocated"] * trim_fraction
        trade = TradeRecord(
            entry_date=pos["entry_date"],
            exit_date=date_str,
            ticker=pos["ticker"],
            market=market,
            direction="long",
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            weight=pos["weight"] * trim_fraction,
            gross_return=gross_ret,
            net_return=net_ret,
            cost=cost,
            exit_reason=f"TRIM:{action.reason}",
            hold_days=int(pos.get("hold_days", 0)) + 1,
            vol_score=pos.get("vol_score", 0),
            opportunity=pos.get("opportunity", 0.0),
        )
        cash_delta = trim_capital * (1 + net_ret)
        pnl_delta = net_ret * pos["weight"] * trim_fraction

        new_positions = list(positions)
        new_positions[idx] = dict(pos)
        new_positions[idx]["weight"] = action.target_weight
        new_positions[idx]["capital_allocated"] = (
            pos["capital_allocated"] * (1.0 - trim_fraction)
        )
        return new_positions, trade, cash_delta, pnl_delta

    def _handle_add_new_action(
        self,
        action: BookAction,
        positions: list[dict],
        today_data: pd.DataFrame,
        vol_scores_today: pd.DataFrame,
        date_str: str,
        cash: float,
        monthly_unique_tickers: set[str],
    ) -> tuple[list[dict], float]:
        """ADD_NEW BookAction → 새 position. (cash_delta는 음수 = 차감)"""
        ticker = action.ticker
        if any(p["ticker"] == ticker for p in positions):
            return positions, 0.0

        tdata = today_data[today_data["ticker"] == ticker]
        if tdata.empty:
            return positions, 0.0
        row = tdata.iloc[0]

        market = self._market_of(ticker)
        ticker_vol = float(row.get("vol_cc_5d", 0.2) or 0.2)
        slippage = self.get_realistic_slippage(market, ticker_vol)
        entry_price = float(row["open"]) * (1 + slippage)
        if entry_price <= 0:
            return positions, 0.0

        weight = action.target_weight
        allocated = cash * weight
        if allocated < self.cfg.trading.min_order_amount_krw:
            return positions, 0.0

        init_unrealized = (
            row["close"] / entry_price - 1.0 if entry_price > 0 else 0.0
        )

        # Lookup vol_score
        vol_score = 0.0
        if (
            not vol_scores_today.empty
            and ticker in vol_scores_today["ticker"].values
        ):
            vol_score = float(
                vol_scores_today[vol_scores_today["ticker"] == ticker]
                ["vol_score"].iloc[0]
            )

        new_pos = {
            "ticker": ticker,
            "entry_date": date_str,
            "entry_price": entry_price,
            "weight": weight,
            "hold_days": 0,
            "entry_vol": ticker_vol,
            "vol_score": vol_score,
            "opportunity": float(action.expected_impact or 0.0),
            "capital_allocated": allocated,
            "last_unrealized": init_unrealized,
        }
        new_positions = list(positions) + [new_pos]
        monthly_unique_tickers.add(ticker)
        return new_positions, -allocated

    def _handle_pyramid_action(
        self,
        action: BookAction,
        positions: list[dict],
        today_data: pd.DataFrame,
        date_str: str,
        cash: float,
    ) -> tuple[list[dict], float]:
        """ADD_TO_WINNER → 기존 position에 추가 capital, weighted-avg entry_price."""
        idx = next(
            (i for i, p in enumerate(positions) if p["ticker"] == action.ticker),
            None,
        )
        if idx is None:
            return positions, 0.0
        pos = positions[idx]
        add_weight = action.target_weight - pos["weight"]
        if add_weight <= 0:
            return positions, 0.0

        tdata = today_data[today_data["ticker"] == action.ticker]
        if tdata.empty:
            return positions, 0.0
        row = tdata.iloc[0]

        market = self._market_of(action.ticker)
        ticker_vol = float(
            row.get("vol_cc_5d", pos.get("entry_vol", 0.2)) or 0.2
        )
        slippage = self.get_realistic_slippage(market, ticker_vol)
        add_entry_price = float(row["open"]) * (1 + slippage)
        if add_entry_price <= 0:
            return positions, 0.0

        add_capital = cash * add_weight
        if add_capital < self.cfg.trading.min_order_amount_krw:
            return positions, 0.0

        # Weighted-average entry_price
        old_capital = pos["capital_allocated"]
        new_capital = old_capital + add_capital
        weighted_entry = (
            (pos["entry_price"] * old_capital + add_entry_price * add_capital)
            / new_capital
        )

        new_positions = list(positions)
        new_positions[idx] = dict(pos)
        new_positions[idx]["weight"] = action.target_weight
        new_positions[idx]["capital_allocated"] = new_capital
        new_positions[idx]["entry_price"] = weighted_entry
        return new_positions, -add_capital

    def _process_book_actions(
        self,
        ctx: DecisionContext,
        positions: list[dict],
        today_data: pd.DataFrame,
        vol_scores_today: pd.DataFrame,
        date_str: str,
        cash: float,
        monthly_unique_tickers: set[str],
    ) -> tuple[list[dict], list[TradeRecord], float, float]:
        """V3.3 path: ctx.actions를 trade + position 변경으로 dispatch.

        Action types handled:
          EXIT, ROTATE → trade + position 제거 (replacement entry는 ADD_NEW로 별도 emit)
          TRIM        → 부분 trade + position weight 감소
          ADD_NEW     → 새 position
          ADD_TO_WINNER → 기존 weight 증가
          KEEP, NO_ACTION, BLOCKED → 변동 없음
        """
        new_positions = list(positions)
        new_trades: list[TradeRecord] = []
        daily_pnl = 0.0

        for action in ctx.actions:
            at = action.action_type
            if at in ("KEEP", "NO_ACTION", "BLOCKED"):
                continue
            if at in ("EXIT", "ROTATE"):
                new_positions, trade, cash_delta, pnl_delta = (
                    self._handle_exit_action(
                        action, new_positions, today_data, date_str,
                    )
                )
                if trade is not None:
                    new_trades.append(trade)
                    cash += cash_delta
                    daily_pnl += pnl_delta
            elif at == "TRIM":
                new_positions, trade, cash_delta, pnl_delta = (
                    self._handle_trim_action(
                        action, new_positions, today_data, date_str,
                    )
                )
                if trade is not None:
                    new_trades.append(trade)
                    cash += cash_delta
                    daily_pnl += pnl_delta
            elif at == "ADD_NEW":
                new_positions, cash_delta = self._handle_add_new_action(
                    action, new_positions, today_data, vol_scores_today,
                    date_str, cash, monthly_unique_tickers,
                )
                cash += cash_delta
                if len(new_positions) >= self.cfg.trading.max_positions:
                    # Skip remaining ADD_NEW actions
                    pass
            elif at == "ADD_TO_WINNER":
                new_positions, cash_delta = self._handle_pyramid_action(
                    action, new_positions, today_data, date_str, cash,
                )
                cash += cash_delta

        # Update unrealized + hold_days for surviving positions
        for pos in new_positions:
            if pos.get("hold_days", 0) == 0:
                # Just-entered ADD_NEW; skip increment (entry day)
                continue
            tdata = today_data[today_data["ticker"] == pos["ticker"]]
            if tdata.empty:
                continue
            row = tdata.iloc[0]
            current_price = float(row["close"])
            unrealized = (
                current_price / pos["entry_price"] - 1.0
                if pos["entry_price"] > 0 else 0.0
            )
            prev_unrealized = pos.get("last_unrealized", unrealized)
            daily_pnl += (unrealized - prev_unrealized) * pos["weight"]
            pos["last_unrealized"] = unrealized
            pos["hold_days"] = pos.get("hold_days", 0) + 1

        return new_positions, new_trades, cash, daily_pnl

    @staticmethod
    def _build_volume_map(today_data: pd.DataFrame) -> dict[str, float]:
        if today_data.empty:
            return {}
        vol_map = {
            str(row["ticker"]): float(row["close"] * row["volume"])
            for _, row in today_data.iterrows()
            if pd.notna(row.get("close")) and pd.notna(row.get("volume"))
        }
        return vol_map

    def _cost_for_market(self, today_data: pd.DataFrame) -> float:
        """Dominant market cost (based on first ticker; usually NASDAQ in V3)."""
        if today_data.empty:
            return self.cfg.trading.costs.us_roundtrip
        first_ticker = str(today_data.iloc[0]["ticker"])
        market = self._market_of(first_ticker)
        return self.cfg.cost_for_market(market)

    def _compute_metrics(
        self, trades: list[TradeRecord], equity_df: pd.DataFrame, capital: float,
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                total_return=0, annual_return=0, sharpe=0, max_drawdown=0,
                win_rate=0, avg_return=0, avg_hold_days=0, total_trades=0,
                avg_trades_per_month=0, total_cost=0, profit_factor=0,
                trades=[], equity_curve=equity_df,
            )

        returns = [t.net_return for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_return = 0.0
        if not equity_df.empty:
            total_return = equity_df["portfolio_value"].iloc[-1] / capital - 1

        if not equity_df.empty and len(equity_df) > 1:
            n_days = len(equity_df)
            annual_factor = 252 / max(n_days, 1)
        else:
            annual_factor = 1.0
        annual_return = (1 + total_return) ** annual_factor - 1

        if not equity_df.empty and "daily_pnl" in equity_df.columns:
            daily_returns = equity_df["daily_pnl"].values
            sharpe = (
                (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                if len(daily_returns) > 5 and daily_returns.std() > 0 else 0.0
            )
        else:
            sharpe = 0.0

        if not equity_df.empty:
            peak = equity_df["portfolio_value"].cummax()
            drawdown = (equity_df["portfolio_value"] - peak) / peak
            max_dd = float(abs(drawdown.min()))
        else:
            max_dd = 0.0

        if trades:
            months = max(
                1,
                (pd.Timestamp(trades[-1].entry_date) - pd.Timestamp(trades[0].entry_date)).days / 30,
            )
            trades_per_month = len(trades) / months
        else:
            trades_per_month = 0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-8
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=len(wins) / max(len(trades), 1),
            avg_return=np.mean(returns),
            avg_hold_days=np.mean([t.hold_days for t in trades]),
            total_trades=len(trades),
            avg_trades_per_month=trades_per_month,
            total_cost=sum(t.cost for t in trades),
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_df,
        )

    @staticmethod
    def _market_of(ticker: str) -> str:
        if ticker.endswith(".KS"):
            return "KOSPI"
        if ticker.endswith(".KQ"):
            return "KOSDAQ"
        if ticker.split(".")[0].isdigit():
            return "KOSPI"
        return "NASDAQ"

    def get_realistic_slippage(self, market: str, ticker_vol: float = 0.2) -> float:
        base = self.cfg.slippage_for_market(market)
        vol_excess = max(0, ticker_vol - 0.20)
        vol_mult = 1.0 + vol_excess * 5.0
        return base * vol_mult
