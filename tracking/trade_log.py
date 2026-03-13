"""거래 기록 및 성과 추적 (SQLite 기반).

매 거래, 일별 수익률, 모델 예측 정확도를 기록합니다.
"""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class TradeLogger:
    """SQLite 기반 거래 로그 및 성과 추적."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        ticker      TEXT NOT NULL,
        side        TEXT NOT NULL,  -- 'buy' or 'sell'
        qty         REAL NOT NULL,
        price       REAL NOT NULL,
        amount      REAL NOT NULL,  -- qty * price
        sector      TEXT,
        order_no    TEXT,
        mode        TEXT,           -- 'sandbox' or 'production'
        note        TEXT
    );

    CREATE TABLE IF NOT EXISTS daily_performance (
        date            TEXT PRIMARY KEY,
        portfolio_value REAL NOT NULL,
        daily_return    REAL NOT NULL,
        benchmark_return REAL,      -- equal-weight 기준선 수익률
        sharpe_30d      REAL,
        mdd_cumul       REAL,
        model_version   TEXT,
        n_trades        INTEGER DEFAULT 0,
        daily_turnover  REAL DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS model_signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        date        TEXT NOT NULL,
        sector      TEXT NOT NULL,
        signal      REAL NOT NULL,  -- 원본 모델 신호
        weight      REAL NOT NULL,  -- 실제 적용 가중치 (alpha blend 포함)
        actual_return REAL          -- 실제 발생한 수익률 (사후 기록)
    );

    CREATE TABLE IF NOT EXISTS retrain_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        phase_start INTEGER NOT NULL,
        duration_sec INTEGER,
        val_loss    REAL,
        dir_acc     REAL,
        trigger     TEXT            -- 'weekly', 'monthly', 'manual'
    );
    """

    def __init__(self, db_path: str = "tracking/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"TradeLogger 초기화: {self.db_path}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            for stmt in self.SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            # Backward compat: add daily_turnover column if missing
            try:
                conn.execute(
                    "ALTER TABLE daily_performance ADD COLUMN daily_turnover REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # 거래 기록
    # ------------------------------------------------------------------

    def log_trade(self, order_result: dict, note: str = "") -> int:
        """주문 결과 기록."""
        qty   = order_result.get("qty", 0)
        price = order_result.get("price", 0)
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (timestamp, ticker, side, qty, price, amount, sector, order_no, mode, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_result.get("timestamp", datetime.now().isoformat()),
                    order_result.get("ticker", ""),
                    order_result.get("side", ""),
                    qty,
                    price,
                    qty * price,
                    order_result.get("sector", ""),
                    order_result.get("order_no", ""),
                    order_result.get("mode", "sandbox"),
                    note,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def log_signals(self, signals: dict[str, float], weights: dict[str, float]):
        """모델 신호 기록 (사후 실제 수익률과 비교용)."""
        today = date.today().isoformat()
        with self._conn() as conn:
            for sector, signal in signals.items():
                conn.execute(
                    """INSERT INTO model_signals (date, sector, signal, weight)
                       VALUES (?,?,?,?)""",
                    (today, sector, signal, weights.get(sector, 0.0)),
                )
            conn.commit()

    def update_actual_returns(self, date_str: str, sector_returns: dict[str, float]):
        """실제 수익률 사후 업데이트 (당일 종가 기준)."""
        with self._conn() as conn:
            for sector, ret in sector_returns.items():
                conn.execute(
                    """UPDATE model_signals SET actual_return=?
                       WHERE date=? AND sector=?""",
                    (ret, date_str, sector),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # 일별 성과 기록
    # ------------------------------------------------------------------

    def log_daily_performance(
        self,
        portfolio_value: float,
        daily_return: float,
        benchmark_return: Optional[float] = None,
        model_version: str = "",
        n_trades: int = 0,
        turnover: float = 0.0,
    ):
        """일별 성과 기록."""
        today = date.today().isoformat()

        # 30일 롤링 Sharpe 계산
        sharpe_30d = self._compute_rolling_sharpe(30)
        mdd_cumul  = self._compute_cumulative_mdd()

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_performance
                   (date, portfolio_value, daily_return, benchmark_return,
                    sharpe_30d, mdd_cumul, model_version, n_trades, daily_turnover)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (today, portfolio_value, daily_return, benchmark_return,
                 sharpe_30d, mdd_cumul, model_version, n_trades, turnover),
            )
            conn.commit()

        logger.info(
            f"일별 성과 기록: date={today}, return={daily_return:.2%}, "
            f"sharpe_30d={sharpe_30d:.2f}, mdd={mdd_cumul:.2%}, turnover={turnover:.2%}"
        )

    def log_retrain(
        self,
        phase_start: int,
        duration_sec: int,
        val_loss: float,
        dir_acc: float,
        trigger: str = "manual",
    ):
        """재학습 이벤트 기록."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO retrain_log
                   (timestamp, phase_start, duration_sec, val_loss, dir_acc, trigger)
                   VALUES (?,?,?,?,?,?)""",
                (datetime.now().isoformat(), phase_start, duration_sec,
                 val_loss, dir_acc, trigger),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # 조회 메서드
    # ------------------------------------------------------------------

    def get_today_pnl_pct(self) -> float:
        """오늘의 수익률 (%) 반환."""
        today = date.today().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT daily_return FROM daily_performance WHERE date=?", (today,)
            ).fetchone()
        return row[0] if row else 0.0

    def get_performance_df(self, days: int = 90) -> pd.DataFrame:
        """최근 N일 성과 DataFrame 반환."""
        with self._conn() as conn:
            df = pd.read_sql(
                f"""SELECT * FROM daily_performance
                    ORDER BY date DESC LIMIT {days}""",
                conn,
            )
        return df.sort_values("date").reset_index(drop=True)

    def get_trades_df(self, days: int = 30) -> pd.DataFrame:
        """최근 N일 거래 내역 DataFrame 반환."""
        with self._conn() as conn:
            df = pd.read_sql(
                f"""SELECT * FROM trades
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC""",
                conn,
            )
        return df

    def get_signal_accuracy(self, days: int = 30) -> dict:
        """모델 신호 정확도 계산 (예측 방향 vs 실제 수익 방향)."""
        with self._conn() as conn:
            df = pd.read_sql(
                f"""SELECT sector, signal, actual_return FROM model_signals
                    WHERE date >= date('now', '-{days} days')
                    AND actual_return IS NOT NULL""",
                conn,
            )

        if df.empty:
            return {"directional_accuracy": None, "n_samples": 0}

        correct = ((df["signal"] > 0) == (df["actual_return"] > 0)).sum()
        return {
            "directional_accuracy": correct / len(df),
            "n_samples": len(df),
            "by_sector": df.groupby("sector").apply(
                lambda g: ((g["signal"] > 0) == (g["actual_return"] > 0)).mean()
            ).to_dict(),
        }

    def get_summary(self) -> dict:
        """전체 운용 요약."""
        perf_df = self.get_performance_df(days=365)
        if perf_df.empty:
            return {"status": "운용 데이터 없음"}

        total_return = (
            perf_df["portfolio_value"].iloc[-1] /
            perf_df["portfolio_value"].iloc[0] - 1
        ) if len(perf_df) > 1 else 0.0

        return {
            "운용기간":      f"{len(perf_df)}일",
            "누적수익률":    f"{total_return:.2%}",
            "30일Sharpe":  f"{perf_df['sharpe_30d'].iloc[-1]:.2f}" if not perf_df.empty else "N/A",
            "최대낙폭":      f"{perf_df['mdd_cumul'].min():.2%}",
            "수익일비율":    f"{(perf_df['daily_return'] > 0).mean():.1%}",
        }

    def compute_turnover(self, date_str: str = None) -> float:
        """일별 턴오버 계산: sum(매수 금액) / 포트폴리오 가치.

        Args:
            date_str: 계산할 날짜 (ISO 형식, None이면 오늘).

        Returns:
            Turnover ratio (0.0 if data is unavailable).
        """
        if date_str is None:
            date_str = date.today().isoformat()

        with self._conn() as conn:
            # 해당 날짜 매수 총액
            row = conn.execute(
                """SELECT COALESCE(SUM(amount), 0)
                   FROM trades
                   WHERE DATE(timestamp) = ? AND side = 'buy'""",
                (date_str,),
            ).fetchone()
            buy_total = row[0] if row else 0.0

            # 해당 날짜 포트폴리오 가치
            pv_row = conn.execute(
                "SELECT portfolio_value FROM daily_performance WHERE date = ?",
                (date_str,),
            ).fetchone()
            portfolio_value = pv_row[0] if pv_row else 0.0

        if portfolio_value <= 0:
            return 0.0
        return buy_total / portfolio_value

    def get_turnover_stats(self, days: int = 30) -> dict:
        """최근 N일 턴오버 통계.

        Args:
            days: 조회 일수.

        Returns:
            dict with:
                mean_daily_turnover: 일평균 턴오버
                annualized_turnover: 연간 환산 턴오버 (mean * 252)
                estimated_annual_cost: 연간 비용 추정 (annualized * 0.006 roundtrip)
        """
        with self._conn() as conn:
            df = pd.read_sql(
                f"""SELECT date, daily_turnover FROM daily_performance
                    ORDER BY date DESC LIMIT {days}""",
                conn,
            )

        if df.empty:
            return {
                "mean_daily_turnover": 0.0,
                "annualized_turnover": 0.0,
                "estimated_annual_cost": 0.0,
            }

        mean_daily = float(df["daily_turnover"].mean())
        annualized = mean_daily * 252
        estimated_cost = annualized * 0.006  # roundtrip cost assumption

        return {
            "mean_daily_turnover": mean_daily,
            "annualized_turnover": annualized,
            "estimated_annual_cost": estimated_cost,
        }

    # ------------------------------------------------------------------
    # 내부 계산
    # ------------------------------------------------------------------

    def _compute_rolling_sharpe(self, window: int = 30) -> float:
        """최근 window일 Sharpe 계산."""
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT daily_return FROM daily_performance
                    ORDER BY date DESC LIMIT {window}""",
            ).fetchall()

        if len(rows) < 5:
            return 0.0

        import numpy as np
        rets = [r[0] for r in rows]
        mean = sum(rets) / len(rets)
        std  = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
        return float(mean / (std + 1e-8) * (252 ** 0.5))

    def _compute_cumulative_mdd(self) -> float:
        """누적 최대낙폭 계산.

        portfolio_value가 0 이하인 레코드는 제외 (데이터 오류 방어).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT portfolio_value FROM daily_performance WHERE portfolio_value > 0 ORDER BY date"
            ).fetchall()

        if not rows:
            return 0.0

        values = [r[0] for r in rows]
        peak = values[0]
        mdd  = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak
            mdd  = min(mdd, dd)
        return mdd
