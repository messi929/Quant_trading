"""Trading runner — 2-session scheduler (v2.2).

KR Session: data(06:00) → signal(06:10) → entry(09:30) → monitor(5min) → close(15:20) → eod(16:00)
US Session: signal(22:00) → entry(23:40) → monitor(5min) → close(04:30)

v2.2 changes:
  - KR entry 09:10 → 09:30 (스프레드 안정화 후 진입)
  - Session close: 만료(3일) 포지션만 청산, 나머지 오버나이트 보유
  - CASH 신호 시 force_close_all
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import schedule
import time as _time

from loguru import logger

from config.config_loader import load_config
from pipeline.signal_pipeline import SignalPipeline
from live.executor import TradingExecutor


class TradingRunner:
    """2-session trading scheduler with 3-day hold strategy."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.pipeline = SignalPipeline(self.cfg)
        self.executor = TradingExecutor(self.cfg)

        self._kr_signal: dict | None = None
        self._us_signal: dict | None = None

        self._kr = self.cfg["schedule"]["kr"]
        self._us = self.cfg["schedule"]["us"]

    # ── Data Collection ────────────────────────────────────────

    def collect_data(self):
        """06:00 — Collect latest market data and update parquet."""
        logger.info("=" * 50)
        logger.info("[DATA] Collecting latest market data")
        try:
            from data.collector import DataCollector
            from data.processor import DataProcessor
            from data.sector_classifier import load_kr_sector_map

            collector = DataCollector(self.cfg)
            end_date = date.today()
            start_date = end_date - timedelta(days=10)

            raw_df = collector.collect_all(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if raw_df is None or len(raw_df) == 0:
                logger.warning("[DATA] No new data collected — using existing parquet")
                return

            processor = DataProcessor(min_history_days=1)
            new_df = processor.process(raw_df)

            static_map = load_kr_sector_map()
            if "sector" not in new_df.columns:
                new_df["sector"] = "unknown"
            for ticker in new_df["ticker"].unique():
                if ticker in static_map:
                    new_df.loc[new_df["ticker"] == ticker, "sector"] = static_map[ticker]

            processed_path = (
                Path(self.cfg["paths"]["processed_data"]) / "processed_data.parquet"
            )
            if processed_path.exists():
                old_df = pd.read_parquet(processed_path)
                if "sector" in old_df.columns:
                    existing_sectors = (
                        old_df[["ticker", "sector"]]
                        .drop_duplicates("ticker")
                        .set_index("ticker")["sector"]
                        .to_dict()
                    )
                    for ticker in new_df["ticker"].unique():
                        if ticker not in static_map and ticker in existing_sectors:
                            new_df.loc[
                                new_df["ticker"] == ticker, "sector"
                            ] = existing_sectors[ticker]

                combined = pd.concat([old_df, new_df]).drop_duplicates(
                    subset=["date", "ticker"], keep="last"
                ).sort_values(["date", "ticker"])
                combined.to_parquet(processed_path, index=False)
                logger.info(
                    f"[DATA] Updated: {len(old_df):,} → {len(combined):,} rows "
                    f"(+{len(combined) - len(old_df):,})"
                )
            else:
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_parquet(processed_path, index=False)
                logger.info(f"[DATA] Created: {len(new_df):,} rows")

        except Exception as e:
            logger.error(f"[DATA] Collection failed: {e}")
            logger.info("[DATA] Will use existing parquet for signal generation")

    # ── KR Session ─────────────────────────────────────────────

    def kr_generate_signal(self):
        """06:10 — Generate KR trading signal."""
        logger.info("=" * 50)
        logger.info("[KR] Signal generation start")
        try:
            self._kr_signal = self.pipeline.generate(market_filter="domestic")
            action = self._kr_signal["action"]
            n_pos = len(self._kr_signal.get("positions", []))
            regime = self._kr_signal.get("regime", {}).get("regime", "unknown")
            logger.info(f"[KR] Signal: {action} | {n_pos} positions | regime={regime}")
        except Exception as e:
            logger.error(f"[KR] Signal generation failed: {e}")
            self._kr_signal = None

    def kr_execute_entry(self):
        """09:30 — Execute KR entry with reconciliation.

        - CASH signal → force close all positions
        - TRADE signal → reconcile (close non-signal, keep matching, buy new)
        """
        logger.info("[KR] Execute entry")
        self.executor.reset_daily()

        if self._kr_signal is None:
            logger.info("[KR] No signal — skipping entry")
            return

        if self._kr_signal["action"] == "CASH":
            logger.info("[KR] CASH — closing all positions")
            self.executor.force_close_all()
            return

        positions = self._kr_signal["positions"]
        results = self.executor.execute_entry(positions)
        logger.info(f"[KR] Entered {len(results)} new positions")

    def kr_monitor(self):
        """Every 5min — Check portfolio risk, profit take, hold expiry."""
        if not self.executor.open_positions:
            return

        exits = self.executor.monitor_positions()
        if exits:
            logger.info(f"[KR] Monitor: {len(exits)} exits triggered")

    def kr_close_session(self):
        """15:20 — Close only expired positions. Keep others overnight."""
        logger.info("[KR] Session close")
        exits = self.executor.execute_exit()
        self._log_daily_performance()

    # ── US Session ─────────────────────────────────────────────

    def us_generate_signal(self):
        """22:00 — Generate FRESH US signal."""
        logger.info("=" * 50)
        logger.info("[US] Signal generation start (fresh)")
        try:
            self._us_signal = self.pipeline.generate(market_filter="overseas")
            action = self._us_signal["action"]
            n_pos = len(self._us_signal.get("positions", []))
            logger.info(f"[US] Signal: {action} | {n_pos} positions")
        except Exception as e:
            logger.error(f"[US] Signal generation failed: {e}")
            self._us_signal = None

    def us_execute_entry(self):
        """23:40 — Execute US entry with reconciliation."""
        logger.info("[US] Execute entry")
        self.executor.reset_daily()

        if self._us_signal is None:
            logger.info("[US] No signal — skipping entry")
            return

        if self._us_signal["action"] == "CASH":
            logger.info("[US] CASH — closing any positions")
            self.executor.force_close_all()
            return

        positions = self._us_signal["positions"]
        results = self.executor.execute_entry(positions)
        logger.info(f"[US] Entered {len(results)} positions")

    def us_monitor(self):
        """Every 5min — Check exits for US positions."""
        if not self.executor.open_positions:
            return

        exits = self.executor.monitor_positions()
        if exits:
            logger.info(f"[US] Monitor: {len(exits)} exits triggered")

    def us_close_session(self):
        """04:30 — Close US positions (paper trading, full close)."""
        logger.info("[US] Session close")
        exits = self.executor.force_close_all()
        logger.info(f"[US] Closed {len(exits)} positions")

    # ── EOD ────────────────────────────────────────────────────

    def _log_daily_performance(self):
        """Log daily performance (includes unrealized P&L from held positions)."""
        try:
            from tracking.trade_log import TradeLogger
            tl = TradeLogger(self.cfg["paths"]["trade_log_db"])

            balance = self.executor.api.get_domestic_balance()
            portfolio_value = balance.get("total_eval", 0)

            perf_df = tl.get_performance_df(days=2)
            if len(perf_df) >= 1:
                prev = float(perf_df["portfolio_value"].iloc[-1])
                daily_return = (portfolio_value - prev) / prev if prev > 0 else 0.0
            else:
                daily_return = 0.0

            n_trades = len(tl.get_trades_df(days=1))
            turnover = tl.compute_turnover(portfolio_value=portfolio_value)

            tl.log_daily_performance(
                portfolio_value=portfolio_value,
                daily_return=daily_return,
                n_trades=n_trades,
                turnover=turnover,
            )

            held = len(self.executor.open_positions)
            logger.info(
                f"EOD: portfolio={portfolio_value:,.0f}, "
                f"return={daily_return:.2%}, trades={n_trades}, "
                f"positions_held={held}"
            )
        except Exception as e:
            logger.error(f"EOD logging failed: {e}")

    # ── Scheduler ──────────────────────────────────────────────

    def setup_schedule(self):
        """Configure schedule from config."""
        kr = self._kr
        us = self._us

        # Data collection
        schedule.every().day.at(kr.get("data_collect", "06:00")).do(self.collect_data)

        # KR session
        schedule.every().day.at(kr["signal_generate"]).do(self.kr_generate_signal)
        schedule.every().day.at(kr["execute_entry"]).do(self.kr_execute_entry)
        schedule.every().day.at(kr["execute_exit"]).do(self.kr_close_session)

        # KR monitor: every N min from entry+interval to exit
        kr_interval = kr.get("monitor_interval_min", 5)
        kr_entry_h, kr_entry_m = map(int, kr["execute_entry"].split(":"))
        kr_exit = kr["execute_exit"]
        cur = kr_entry_h * 60 + kr_entry_m + kr_interval
        while True:
            t = f"{cur // 60:02d}:{cur % 60:02d}"
            if t > kr_exit:
                break
            schedule.every().day.at(t).do(self.kr_monitor)
            cur += kr_interval

        # EOD
        schedule.every().day.at(kr.get("eod_record", "16:00")).do(
            self._log_daily_performance
        )

        # US session
        schedule.every().day.at(us["signal_generate"]).do(self.us_generate_signal)
        schedule.every().day.at(us["execute_entry"]).do(self.us_execute_entry)
        schedule.every().day.at(us["execute_exit"]).do(self.us_close_session)

        # US monitor
        us_interval = us.get("monitor_interval_min", 5)
        us_entry_h, us_entry_m = map(int, us["execute_entry"].split(":"))
        us_exit_h, us_exit_m = map(int, us["execute_exit"].split(":"))
        us_exit_abs = us_exit_h * 60 + us_exit_m
        cur = (us_entry_h * 60 + us_entry_m + us_interval) % (24 * 60)
        while True:
            t = f"{cur // 60:02d}:{cur % 60:02d}"
            if cur <= us_exit_abs:
                schedule.every().day.at(t).do(self.us_monitor)
                if cur == us_exit_abs:
                    break
                cur += us_interval
                if cur > us_exit_abs:
                    break
            else:
                schedule.every().day.at(t).do(self.us_monitor)
                cur = (cur + us_interval) % (24 * 60)

        logger.info("Schedule configured:")
        logger.info(
            f"  KR: signal={kr['signal_generate']}, "
            f"entry={kr['execute_entry']}, exit={kr['execute_exit']}"
        )
        logger.info(
            f"  US: signal={us['signal_generate']}, "
            f"entry={us['execute_entry']}, exit={us['execute_exit']}"
        )

    def run_daemon(self):
        """Run the scheduler loop."""
        self.setup_schedule()
        logger.info("Trading daemon started. Press Ctrl+C to stop.")

        while True:
            try:
                schedule.run_pending()
                _time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                _time.sleep(60)

    # ── Manual execution ───────────────────────────────────────

    def run_kr_session_now(self):
        """Manually run a full KR session (for testing)."""
        self.kr_generate_signal()
        self.kr_execute_entry()
        self.kr_monitor()
        self.kr_close_session()

    def run_us_session_now(self):
        """Manually run a full US session (for testing)."""
        self.us_generate_signal()
        self.us_execute_entry()
        self.us_monitor()
        self.us_close_session()
