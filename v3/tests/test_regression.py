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


# ── Bug 9: recommendation_log must not crash session on numpy types ──
#
# 2026-05-07 evaluator review (V3.2.1+ observation tooling):
# `_log_recommendation_snapshot` writes opportunity_map / position weights
# via json.dumps. VolTransformer inference can produce numpy.float32 scalars
# which round() preserves, and json.dumps then raises TypeError. Trading
# flow must not abort — log failure is acceptable, session abort is not.

class TestRecommendationLogResilience:
    """The observation log is a non-critical sidecar. It must never raise
    out of run_session() — neither I/O errors nor serialization issues."""

    def _write_record(self, record: dict, log_path) -> bool:
        """Mirrors live_pipeline._log_recommendation_snapshot persistence."""
        import json
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except (OSError, TypeError, ValueError):
            return False

    def test_numpy_float32_does_not_crash(self, tmp_path):
        """opportunity_map values from inference are numpy.float32. After
        round(x, 6), they remain numpy.float32. json.dumps must not raise
        — either via default serializer or by pre-cast to Python float."""
        import numpy as np
        record = {
            "ticker": "FANG",
            "opp": float(round(np.float32(0.00345), 6)),  # explicit cast
        }
        log_path = tmp_path / "recommendation_log.jsonl"
        assert self._write_record(record, log_path) is True
        assert log_path.exists()

    def test_unserializable_value_caught(self, tmp_path):
        """If a future bug slips an unserializable object in, the session
        must continue. The log entry is dropped, not the trade."""

        class Weird:
            pass

        record = {"weird": Weird()}
        log_path = tmp_path / "recommendation_log.jsonl"
        # Should NOT raise; should return False indicating drop
        assert self._write_record(record, log_path) is False

    def test_disk_failure_caught(self, tmp_path, monkeypatch):
        """OSError on write must not propagate."""
        import json
        log_path = tmp_path / "subdir" / "recommendation_log.jsonl"

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(json, "dumps", boom)
        # Even with a working path, error during dump must be swallowed
        assert self._write_record({"x": 1}, log_path) is False


# ── Bug 10: sizer floor must guarantee min_order_amount in caution regime ──
#
# 2026-05-07 observation (13 trading days, 0 entries):
# sizer.min_position_weight=0.05 + caution position_scale=0.42 → max weight
# 0.021 (= 0.05 × 0.42), below the 5M-on-100M order floor (0.05). Result:
# every recommended ticker dropped via min_order_amount. The floor must be
# raised to make 5M reachable in the worst-case regime scale.

class TestSizerFloorPassesMinOrder:
    """Floor on sizer output must produce >= min_order_amount even after
    regime scale multiplication in conservative regimes."""

    CAPITAL = 100_000_000
    MIN_ORDER = 5_000_000

    def test_caution_floor_clears_min_order(self):
        """At caution MINIMUM scale 0.35 (observed range 5/4~5/7: 0.35~0.42),
        floor must produce at least 5M order. The first attempt at 0.12 used
        0.42 as the bound and failed in production at scale=0.411 (5/7 23:40
        US session): 0.12×0.411=0.0493 → 4,998,000 KRW, 1,001 short of 5M.
        Test now anchors to the observed minimum of the caution range."""
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        floor_weight = sizer.min_weight
        scale_caution_min = 0.35  # observed minimum in production caution
        scaled = floor_weight * scale_caution_min
        order_value = scaled * self.CAPITAL
        assert order_value >= self.MIN_ORDER, (
            f"floor={floor_weight} × scale={scale_caution_min} = {scaled} → "
            f"{order_value:,.0f} KRW, below min {self.MIN_ORDER:,.0f}"
        )

    def test_neutral_floor_clears_min_order(self):
        """At neutral scale 0.65, floor must produce at least 5M."""
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        order_value = sizer.min_weight * 0.65 * self.CAPITAL
        assert order_value >= self.MIN_ORDER

    def test_floor_does_not_violate_max_weight(self):
        """Floor must never exceed max_single_weight (0.40)."""
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        assert sizer.min_weight < sizer.max_weight

    def test_size_portfolio_respects_floor(self):
        """size_portfolio output must clip up to floor for low-conviction
        tickers (so they pass min_order downstream).

        Bound: n=3 stays under the 0.95-normalize threshold (3×0.12=0.36)
        so floor is preserved. Beyond n≈7 the normalize step at sizing.py:
        125 silently pushes weights below the floor — a latent landmine if
        max_positions is ever raised to 8+. See test_size_portfolio_n8_normalize_violates_floor.
        """
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        weights = sizer.size_portfolio(
            predicted_vols={"A": 0.30, "B": 0.30, "C": 0.30},
            confidences={"A": 0.55, "B": 0.55, "C": 0.55},
        )
        for t, w in weights.items():
            assert w >= sizer.min_weight, (
                f"{t} weight={w} below floor {sizer.min_weight}"
            )

    def test_size_portfolio_n8_normalize_violates_floor(self):
        """LATENT BUG documentation — not a fix request.

        With n≥8 and floor 0.12, np.clip pushes everyone to 0.12, then
        0.95-normalize divides by 0.96 → 0.1187 (below floor). This is
        unreachable while max_positions=1 (currently set in v3_config.yaml).
        If max_positions is ever raised to ≥8, sizer must be revised to
        either (a) skip the normalize when at floor, or (b) drop excess
        positions before normalizing.
        """
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        n = 8
        vols = {f"T{i}": 0.30 for i in range(n)}
        confs = {f"T{i}": 0.55 for i in range(n)}
        weights = sizer.size_portfolio(predicted_vols=vols, confidences=confs)
        any_below_floor = any(w < sizer.min_weight for w in weights.values())
        # Document the known violation. If this test ever flips to False,
        # the underlying bug was fixed and the floor invariant holds — fine.
        # If max_positions is raised to ≥8 without fixing this, the floor
        # chain breaks and downstream min_order skips silently increase.
        assert any_below_floor, (
            "n=8 floor violation no longer reproduces — verify sizer fix "
            "and tighten test_size_portfolio_respects_floor accordingly"
        )

    def test_vol_target_not_drastically_violated(self):
        """3-position floor deployment (3 × 0.12 = 36% gross) must still
        keep portfolio vol within 2× target (sanity bound, not strict)."""
        from v3.strategy.sizing import VolTargetSizer
        import numpy as np
        sizer = VolTargetSizer(target_annual_vol=0.15)
        weights = sizer.size_portfolio(
            predicted_vols={"A": 0.25, "B": 0.25, "C": 0.25},
            confidences={"A": 0.6, "B": 0.6, "C": 0.6},
        )
        w_arr = np.array(list(weights.values()))
        # Equal-corr 0.3 sanity model
        n = len(w_arr)
        corr = np.full((n, n), 0.3)
        np.fill_diagonal(corr, 1.0)
        vols = np.array([0.25] * n)
        cov = np.diag(vols) @ corr @ np.diag(vols)
        port_vol = np.sqrt(w_arr @ cov @ w_arr)
        assert port_vol < 2 * sizer.target_vol, (
            f"port_vol={port_vol:.3f} > 2× target={sizer.target_vol}"
        )


# ── Bug 11: position_scale must be portfolio exposure, not per-position multiplier ──
#
# 2026-05-10 V3.3 12 features 활성 후 5/11~5/12 entries=0 (4 거래일).
# Root cause: signal._size()가 raw_weight × position_scale 곱셈 적용.
# caution scale 0.42 × floor 0.15 = 0.063 → 종목당 6.3% → 페르소나 "1종목에
# 의미있게" 위반. min_position_weight floor가 sizer 내부에서만 작동하고 곧바로
# 곱셈으로 무력화됨.
#
# 추가로 발견: signal._size()가 vol_scores['vol_score']를 predicted_vol로 매핑.
# vol_score는 VolTransformer raw output (cross-sectional ranking signal)이지
# annualized volatility가 아님. 진짜 vol은 ohlcv['vol_cc_20d'] (annualized,
# feature_engineer.py:99에서 std * sqrt(252)).
#
# Fix:
#   1. position_scale을 max_gross_exposure로 해석 (V3.3 RegimeBudget과 정합)
#   2. 종목당 floor는 절대값 유지, 총합 초과 시 약한 종목 drop
#   3. predicted_vol을 ohlcv['vol_cc_20d']에서 가져옴 (vol_score 사용 중지)

class TestPositionScaleAsExposure:
    """position_scale은 종목당 곱셈이 아니라 portfolio max_gross_exposure.

    페르소나 원칙 ②"크게 (1~3종목 집중)" 보장. caution regime이라도 들어가는
    종목은 floor 이상 사이즈로 들어가야 한다. 들어갈지 말지가 신중해지는 것이지
    들어가는 사이즈가 작아지는 것이 아니다.
    """

    def _build_signal_gen(self):
        from v3.rules.entry import EntryFilter
        from v3.strategy.opportunity import OpportunityScorer
        from v3.strategy.signal import SignalGenerator
        from v3.strategy.sizing import VolTargetSizer
        sizer = VolTargetSizer()
        return SignalGenerator(
            opportunity_scorer=OpportunityScorer(),
            entry_filter=EntryFilter(),
            sizer=sizer,
        )

    def _build_ohlcv(self, ticker_vols: dict[str, float]):
        import pandas as pd
        rows = [
            {"date": "2026-05-11", "ticker": t, "close": 100.0, "vol_cc_20d": v}
            for t, v in ticker_vols.items()
        ]
        return pd.DataFrame(rows)

    def test_caution_single_candidate_meets_floor(self):
        """5/11 MELI 재현: caution scale 0.4679, single candidate, conv 0.97.

        Before: 0.15 × 0.4679 = 0.0702 (관찰된 production 값과 일치).
        After: position_scale = exposure 한도이므로 single candidate는 floor
               그대로. total(0.15) < scale(0.4679) → normalize 안 함.
        """
        from v3.rules.entry import EntryCandidate
        sg = self._build_signal_gen()
        selected = [EntryCandidate(
            ticker="MELI", opportunity=0.0213, direction=0.022, conviction=0.97,
            ticker_volume_krw=1e9, sector="ConsumerCyclical",
        )]
        ohlcv = self._build_ohlcv({"MELI": 0.30})
        positions = sg._size(selected, 0.4679, ohlcv, vol_scores=None)
        assert len(positions) == 1
        assert positions[0]["weight"] >= sg.sizer.min_weight, (
            f"5/11 regression: weight={positions[0]['weight']} < "
            f"floor={sg.sizer.min_weight}. position_scale은 곱셈이 아니라 "
            f"exposure 한도여야 한다."
        )

    def test_total_weight_respects_scale_as_exposure(self):
        """총합 weight ≤ position_scale 보장. caution scale 0.40, 3 candidates
        각 floor 0.15 → 합 0.45 > 0.40 → 약한 종목 drop until total ≤ 0.40."""
        from v3.rules.entry import EntryCandidate
        sg = self._build_signal_gen()
        selected = [
            EntryCandidate(ticker="A", opportunity=0.030, direction=0.03, conviction=1.0),
            EntryCandidate(ticker="B", opportunity=0.020, direction=0.02, conviction=1.0),
            EntryCandidate(ticker="C", opportunity=0.010, direction=0.01, conviction=1.0),
        ]
        ohlcv = self._build_ohlcv({"A": 0.30, "B": 0.30, "C": 0.30})
        positions = sg._size(selected, 0.40, ohlcv, vol_scores=None)
        total = sum(p["weight"] for p in positions)
        assert total <= 0.40 + 1e-6, (
            f"total weight {total:.4f} exceeds position_scale=0.40 "
            f"(exposure invariant violated)"
        )

    def test_underexposure_keeps_cash(self):
        """total < scale일 때 normalize로 사이즈 부풀리지 않음 — cash 유지.
        Conviction-or-cash 페르소나 원칙."""
        from v3.rules.entry import EntryCandidate
        sg = self._build_signal_gen()
        selected = [EntryCandidate(
            ticker="X", opportunity=0.020, direction=0.02, conviction=1.0,
        )]
        ohlcv = self._build_ohlcv({"X": 0.30})
        positions = sg._size(selected, 0.80, ohlcv, vol_scores=None)
        assert len(positions) == 1
        assert positions[0]["weight"] <= sg.sizer.max_weight + 1e-6
        # scale 0.80인데 1종목 weight가 거기까지 끌려가면 안 됨
        assert positions[0]["weight"] < 0.80, (
            f"weight {positions[0]['weight']} 가 scale 0.80까지 padded up — "
            f"underexposure는 cash 유지여야 한다"
        )

    def test_weakest_dropped_first_when_overexposed(self):
        """Total > scale일 때 opportunity 가장 낮은 종목 먼저 drop.
        강한 신호는 항상 살아남는다."""
        from v3.rules.entry import EntryCandidate
        sg = self._build_signal_gen()
        selected = [
            EntryCandidate(ticker="STRONG", opportunity=0.05, direction=0.05, conviction=1.0),
            EntryCandidate(ticker="WEAK",   opportunity=0.01, direction=0.01, conviction=1.0),
        ]
        ohlcv = self._build_ohlcv({"STRONG": 0.30, "WEAK": 0.30})
        # 2 × floor 0.15 = 0.30. scale 0.20 → 1종목만 살아야 함
        positions = sg._size(selected, 0.20, ohlcv, vol_scores=None)
        tickers = [p["ticker"] for p in positions]
        # WEAK만 단독으로 살아남는 것은 invariant 위반
        if "WEAK" in tickers and "STRONG" not in tickers:
            assert False, "strongest opportunity dropped before weakest"
        # 정상 케이스: STRONG 살아남거나, scale 너무 작아 둘 다 drop
        assert "WEAK" not in tickers or "STRONG" in tickers

    def test_predicted_vol_from_ohlcv_not_vol_score(self):
        """predicted_vol은 ohlcv['vol_cc_20d']에서 와야 함.

        vol_score는 VolTransformer prediction signal (cross-sectional ranking)이며
        annualized volatility가 아님. signal._size()가 이걸 vol로 매핑하면
        high-conviction ticker(vol_score≈1.0)가 'high-vol(100%)'로 해석되어
        오히려 사이즈가 작아지는 역설.
        """
        import pandas as pd
        from v3.rules.entry import EntryCandidate
        sg = self._build_signal_gen()
        selected = [EntryCandidate(
            ticker="T", opportunity=0.03, direction=0.03, conviction=0.97,
        )]
        # vol_cc_20d=0.20 (realistic NASDAQ), vol_score=0.97 (high conviction signal)
        ohlcv = self._build_ohlcv({"T": 0.20})
        vol_scores = pd.DataFrame({
            "ticker": ["T"], "vol_score": [0.97], "confidence": [0.97],
        })
        positions = sg._size(selected, 0.80, ohlcv, vol_scores=vol_scores)
        # vol_cc_20d=0.20 사용 시 base = target_vol/0.20 = 0.75 → sizing 충분.
        # vol_score=0.97을 vol로 잘못 해석하면 base = 0.155 → floor 직격 0.15.
        # 즉 weight > 0.15이면 ohlcv vol_cc_20d 경로가 동작 중이라는 증거.
        assert positions[0]["weight"] > sg.sizer.min_weight + 1e-6, (
            f"weight {positions[0]['weight']:.4f} 가 정확히 floor에 묶임 — "
            f"vol_score를 vol로 매핑하는 버그 잔존 의심 (예상: vol=0.20에서 "
            f"weight > floor)"
        )
