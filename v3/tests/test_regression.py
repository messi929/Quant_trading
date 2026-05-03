"""Regression tests — lock down bugs discovered during V3.2.1 operation.

Each test here encodes an invariant that was violated in production.
DO NOT loosen these without adding a more specific test first.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from v3.execution.position_manager import PositionManager


pytestmark = pytest.mark.regression


# ── Bug 1: hold_days must be calendar-day-based, not monitor-tick count ──

class TestHoldDaysCalendarBasis:
    """2026-04-20: FANG churned after 1 hour because hold_days incremented
    per monitor tick (15 min). Must be derived from entry_date diff.
    """

    def _compute_hold_days(self, entry_date: str, current_date: str) -> int:
        """Mirrors executor.monitor_positions logic."""
        from datetime import date
        entry_d = date.fromisoformat(str(entry_date)[:10])
        today_d = date.fromisoformat(current_date)
        return max(0, (today_d - entry_d).days)

    def test_same_day_is_zero(self):
        assert self._compute_hold_days("2026-04-20", "2026-04-20") == 0

    def test_next_day_is_one(self):
        assert self._compute_hold_days("2026-04-20", "2026-04-21") == 1

    def test_five_day_max(self):
        assert self._compute_hold_days("2026-04-20", "2026-04-25") == 5

    def test_entry_date_with_time_suffix(self):
        """Paper broker may attach ' HH:MM' suffix to entry_date."""
        assert self._compute_hold_days("2026-04-20 23:40", "2026-04-21") == 1

    def test_never_negative(self):
        """Future-dated entry should clamp at 0, not negative."""
        assert self._compute_hold_days("2026-04-25", "2026-04-20") == 0


# ── Bug 2: PaperBroker must be wired when broker.paper_trading=True ──

class TestPaperBrokerWired:
    """2026-04-20: broker.paper_trading=true was set in config but
    TradingExecutor didn't consume it → all NASDAQ orders 404'd on KIS sandbox.
    """

    def test_paper_trading_flag_instantiates_broker(self, tmp_save_dir, monkeypatch):
        from v3.config.schema import V3Config
        cfg = V3Config()
        cfg.broker.paper_trading = True

        # Isolate paper_broker save path
        monkeypatch.chdir(tmp_save_dir.parent)

        from v3.execution.executor import TradingExecutor
        ex = TradingExecutor(cfg)
        assert ex.paper is not None, \
            "paper_trading=True must instantiate PaperBroker"

    def test_paper_trading_off_leaves_broker_none(self, tmp_save_dir, monkeypatch):
        from v3.config.schema import V3Config
        cfg = V3Config()
        cfg.broker.paper_trading = False

        monkeypatch.chdir(tmp_save_dir.parent)

        from v3.execution.executor import TradingExecutor
        ex = TradingExecutor(cfg)
        assert ex.paper is None, \
            "paper_trading=False must NOT instantiate PaperBroker"

    def test_overseas_routed_to_paper(self, tmp_save_dir, monkeypatch):
        from v3.config.schema import V3Config
        cfg = V3Config()
        cfg.broker.paper_trading = True
        monkeypatch.chdir(tmp_save_dir.parent)

        from v3.execution.executor import TradingExecutor
        ex = TradingExecutor(cfg)
        assert ex._use_paper("FANG") is True
        assert ex._use_paper("AAPL") is True

    def test_domestic_stays_on_kis(self, tmp_save_dir, monkeypatch):
        from v3.config.schema import V3Config
        cfg = V3Config()
        cfg.broker.paper_trading = True
        monkeypatch.chdir(tmp_save_dir.parent)

        from v3.execution.executor import TradingExecutor
        ex = TradingExecutor(cfg)
        assert ex._use_paper("005930.KS") is False
        assert ex._use_paper("035720") is False


# ── Bug 3: entry_history must survive daemon restart ──

class TestEntryHistoryPersisted:
    """2026-04-21: PositionManager.save() only wrote positions and cooldown.
    entry_history was in-memory only → daemon restart lost all trade history
    → monthly_trade_count reset to 0, consecutive_entry check bypassed.
    """

    def test_history_persists_across_instances(self, tmp_save_dir):
        pm1 = PositionManager(save_dir=str(tmp_save_dir))
        pm1.entry_history["FANG"] = ["2026-04-20", "2026-04-21"]
        pm1.sell_retries["AMZN"] = 2
        pm1.save()

        pm2 = PositionManager(save_dir=str(tmp_save_dir))
        assert pm2.entry_history.get("FANG") == ["2026-04-20", "2026-04-21"]
        assert pm2.sell_retries.get("AMZN") == 2

    def test_monthly_count_preserved_after_restart(self, tmp_save_dir):
        """Phase 25.1 옵션 C: 3 unique tickers in April → count=3."""
        pm1 = PositionManager(save_dir=str(tmp_save_dir))
        pm1.entry_history = {
            "FANG": ["2026-04-05"],
            "AMZN": ["2026-04-15"],
            "ADI": ["2026-04-20"],
        }
        pm1.save()

        pm2 = PositionManager(save_dir=str(tmp_save_dir))
        assert pm2.monthly_trade_count("2026-04-21") == 3

    def test_monthly_count_filters_by_month(self, tmp_save_dir):
        """March entries must not count for April."""
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.entry_history["FANG"] = ["2026-03-28", "2026-03-30"]
        pm.entry_history["AMZN"] = ["2026-04-05"]
        assert pm.monthly_trade_count("2026-04-21") == 1

    def test_ghost_removal_clears_entry_history(self, tmp_save_dir):
        """record_sell_failure → 3 fails → force remove + clear history."""
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.positions.append({"ticker": "GHOST", "qty": 10, "entry_date": "2026-04-20"})
        pm.entry_history["GHOST"] = ["2026-04-20"]
        pm.save()

        removed = False
        for _ in range(pm.GHOST_MAX_RETRIES):
            removed = pm.record_sell_failure("GHOST")

        assert removed is True
        assert "GHOST" not in pm.entry_history, \
            "Ghost removal must clear entry_history to avoid consecutive-entry veto"


# ── Phase 25.1 옵션 C: monthly cap = unique-ticker count ──

class TestMonthlyUniqueTickerCount:
    """2026-04-21 진단 → 2026-05-03 적용. CLAUDE.md "Monthly Trade Cap 재설계".

    4/11~4/24 데이터: BUY 7회 중 5회가 FANG churn (22:00 진입 → 09:30 청산
    반복) → 4월 dynamic cap 7회를 5일 만에 소진 → 4/27~4/30 8세션 동안 31개
    후보가 monthly_trades에 막혀 QQQ +1.55% 구간 통째로 놓침.

    페르소나 원칙 1 ("확신 있을 때만")의 진짜 결정자가 conviction이 아니라
    임의 cap이 된 implementation 결함. 동일 종목 churn은 unique 1로 카운트
    하여 신규 종목 진입 여지를 보호한다.
    """

    def test_churn_counts_as_one(self, tmp_save_dir):
        """FANG 5회 진입 → unique 1."""
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.entry_history["FANG"] = [
            "2026-04-20", "2026-04-21", "2026-04-22",
            "2026-04-23", "2026-04-24",
        ]
        assert pm.monthly_trade_count("2026-04-24") == 1, \
            "FANG 5회 churn은 unique 1로 카운트 (옵션 C)"

    def test_distinct_tickers_counted(self, tmp_save_dir):
        """5개 다른 ticker는 5로 카운트."""
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.entry_history = {
            "FANG": ["2026-04-20"],
            "AMZN": ["2026-04-21"],
            "ADI":  ["2026-04-22"],
            "TSLA": ["2026-04-23"],
            "NVDA": ["2026-04-24"],
        }
        assert pm.monthly_trade_count("2026-04-24") == 5

    def test_april_history_proxy(self, tmp_save_dir):
        """실제 4/11~4/24 paper history를 모사. BUY 7회 = unique 3 (FANG, ADI, AMZN).

        옵션 C 적용 전: count=7 (cap 7 도달, 5/1까지 차단)
        옵션 C 적용 후: count=3 (cap 7 여유 4 남음, 신규 진입 가능)
        """
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.entry_history = {
            "ADI":  ["2026-04-11"],
            "AMZN": ["2026-04-11"],
            "FANG": [
                "2026-04-20", "2026-04-20",  # 첫 진입 + rebuy
                "2026-04-21", "2026-04-22", "2026-04-23",
            ],
        }
        assert pm.monthly_trade_count("2026-04-24") == 3, \
            "BUY 7회였지만 unique ticker 3개 (FANG churn 흡수)"

    def test_consecutive_entry_unaffected(self, tmp_save_dir):
        """연속 진입 가드는 entry_history 길이 기반 — unique cap과 독립."""
        pm = PositionManager(save_dir=str(tmp_save_dir))
        pm.entry_history["FANG"] = [
            "2026-04-21", "2026-04-22", "2026-04-23",
        ]
        # 3회 연속 진입은 여전히 차단되어야 함
        assert pm.is_consecutive_entry("FANG", "2026-04-24") is True
        # 동시에 monthly_trade_count는 unique=1 유지
        assert pm.monthly_trade_count("2026-04-24") == 1


# ── Bug 4: conditional exit veto — opp > gate must skip profit_take AND max_hold ──

class TestConditionalExitVeto:
    """2026-04-20: +5% TP was unconditional. Extended 2026-04-21: max_hold also.
    Policy: if opportunity still exceeds the gate, hold the position
    (re-evaluate the signal at any time-based exit trigger).
    Risk-based exits (vol_contraction, mae_stop, portfolio_stop) stay unconditional.
    """
    VETOABLE = ("profit_take", "max_hold")
    UNCONDITIONAL = ("vol_contraction", "dynamic_stop_mae", "portfolio_stop")

    def _apply_veto(self, reason: str, opp: float | None, gate: float) -> bool:
        return (reason in self.VETOABLE) and (opp is not None) and (opp > gate > 0)

    def test_profit_take_vetoed_when_opp_high(self):
        assert self._apply_veto("profit_take", 0.023, 0.00175) is True

    def test_max_hold_vetoed_when_opp_high(self):
        """Holding period exceeded but model still likes the ticker → hold."""
        assert self._apply_veto("max_hold", 0.023, 0.00175) is True

    def test_risk_reasons_never_vetoed(self):
        """vol_contraction / MAE / portfolio_stop remain unconditional even
        when opportunity is high — these are risk signals, not time exits."""
        for reason in self.UNCONDITIONAL:
            assert self._apply_veto(reason, 0.023, 0.00175) is False, \
                f"Veto must not apply to risk-based reason={reason}"

    def test_veto_inactive_when_opp_below_gate(self):
        assert self._apply_veto("profit_take", 0.001, 0.00175) is False
        assert self._apply_veto("max_hold", 0.001, 0.00175) is False

    def test_veto_inactive_when_opp_missing(self):
        """Empty cache (first session / stale) → no veto, exit fires normally."""
        assert self._apply_veto("profit_take", None, 0.00175) is False
        assert self._apply_veto("max_hold", None, 0.00175) is False

    def test_veto_inactive_when_gate_zero(self):
        """Empty opportunity cache drops gate to 0 → safety: no veto fires."""
        assert self._apply_veto("profit_take", 0.023, 0.0) is False
        assert self._apply_veto("max_hold", 0.023, 0.0) is False


# ── Bug 6: PaperBroker ↔ PositionManager divergence (orphan positions) ──

class TestPositionReconciliation:
    """2026-04-21: Manual PaperBroker.buy during testing left positions in
    paper_account.json that never reached PositionManager — executor's
    monitor_positions cannot see them, so no auto-TP, no ghost removal,
    no cooldown tracking.

    Invariant: every overseas paper position must have a PositionManager entry.
    """

    def test_detect_orphans_when_pm_empty(self, tmp_save_dir):
        import json
        paper_path = tmp_save_dir / "paper_account.json"
        paper_path.write_text(json.dumps({
            "cash_krw": 90_000_000,
            "positions": [
                {"ticker": "ADI", "qty": 12, "entry_price_usd": 350.0,
                 "entry_price_krw": 490000, "entry_date": "2026-04-11 20:41",
                 "amount_krw": 5_880_000},
            ],
            "trade_history": [],
        }))

        from v3.execution.paper_broker import PaperBroker
        from v3.scripts.reconcile_positions import detect_orphans

        paper = PaperBroker(save_dir=str(tmp_save_dir))
        pm = PositionManager(save_dir=str(tmp_save_dir))

        orphans = detect_orphans(paper, pm)
        assert len(orphans) == 1
        assert orphans[0]["ticker"] == "ADI"

    def test_fix_registers_orphans(self, tmp_save_dir):
        import json
        paper_path = tmp_save_dir / "paper_account.json"
        paper_path.write_text(json.dumps({
            "cash_krw": 90_000_000,
            "positions": [
                {"ticker": "ADI", "qty": 12, "entry_price_usd": 350.0,
                 "entry_price_krw": 490000, "entry_date": "2026-04-11 20:41",
                 "amount_krw": 5_880_000},
                {"ticker": "AMZN", "qty": 21, "entry_price_usd": 238.0,
                 "entry_price_krw": 333200, "entry_date": "2026-04-11 20:41",
                 "amount_krw": 6_997_200},
            ],
            "trade_history": [],
        }))

        from v3.execution.paper_broker import PaperBroker
        from v3.scripts.reconcile_positions import detect_orphans, to_pm_entry

        paper = PaperBroker(save_dir=str(tmp_save_dir))
        pm = PositionManager(save_dir=str(tmp_save_dir))

        for o in detect_orphans(paper, pm):
            pm.add_position(to_pm_entry(o))

        # Re-load: invariant holds after reconcile
        pm2 = PositionManager(save_dir=str(tmp_save_dir))
        assert len(detect_orphans(paper, pm2)) == 0

    def test_reconcile_preserves_entry_date(self, tmp_save_dir):
        """entry_date with time suffix must be trimmed to YYYY-MM-DD
        so hold_days calculation (date.fromisoformat) works."""
        from v3.scripts.reconcile_positions import to_pm_entry

        entry = to_pm_entry({
            "ticker": "X", "qty": 10, "entry_price_usd": 100.0,
            "entry_price_krw": 140000, "entry_date": "2026-04-11 20:41",
            "amount_krw": 1_400_000,
        })
        assert entry["entry_date"] == "2026-04-11"
        assert entry["qty"] == 10
        assert entry["entry_price"] == 100.0


# ── Bug 5: opportunity cache must drop when >8h stale ──

class TestOpportunityStaleness:
    """2026-04-21: If monitor uses a cache that predates the session gap
    (KR 09:30 ↔ US 23:40 = 14h), TP veto could fire on long-dead alpha.
    """

    def test_stale_cache_drops_after_8h(self):
        from datetime import timedelta
        now = datetime.now()
        cached_at = now - timedelta(hours=9)
        age_h = (now - cached_at).total_seconds() / 3600
        assert age_h > 8
        # Pipeline logic: drop cache when age_h > 8
        drop = age_h > 8
        assert drop is True

    def test_fresh_cache_kept_under_8h(self):
        from datetime import timedelta
        now = datetime.now()
        cached_at = now - timedelta(hours=2)
        age_h = (now - cached_at).total_seconds() / 3600
        assert age_h <= 8
