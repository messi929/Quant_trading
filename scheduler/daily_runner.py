"""일간 자동 실행 스케줄러.

매일 아침 실행 순서:
  06:00 - 데이터 수집 (yfinance + pykrx)
  06:30 - 모델 추론 → 섹터 신호 생성 + 하락 신호 즉시 매도
  09:10 - TWAP Wave 1 (40%) — 장 시작 10분 후
  11:00 - TWAP Wave 2 (35%) — 오전 중반
  13:30 - TWAP Wave 3 (25%) — 오후 장
  16:00 - 종가 기록 + 성과 업데이트

실행 방법:
  python scheduler/daily_runner.py           # 즉시 전체 실행 (Wave 1)
  python scheduler/daily_runner.py --step collect       # 데이터 수집만
  python scheduler/daily_runner.py --step signal        # 신호 생성만
  python scheduler/daily_runner.py --step order         # 주문만 (Wave 1)
  python scheduler/daily_runner.py --step order --wave 2  # Wave 2만
  python scheduler/daily_runner.py --step eod           # 종가 기록만
  python scheduler/daily_runner.py --daemon             # 스케줄러 데몬 실행
"""

import sys
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from utils.logger import setup_logger
from utils.device import set_seed


class DailyRunner:
    """일간 자동 실행 파이프라인."""

    def __init__(self, config_path: str = "config/live_config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        with open(self.cfg["model"]["config_path"], encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

        setup_logger(
            log_dir=self.settings["paths"]["logs"],
            level=self.settings["logging"]["level"],
        )
        set_seed(self.settings["training"]["seed"])

        self.execution_market = self.cfg["trading"]["execution_market"]
        self.alpha_blend = 0.4  # 40% model + 60% equal-weight (최적 설정)

        logger.info("DailyRunner 초기화 완료")

    # ------------------------------------------------------------------
    # Step 1: 데이터 수집
    # ------------------------------------------------------------------

    def step_collect(self):
        """최신 시장 데이터 수집 및 전처리."""
        logger.info("=== Step 1: 데이터 수집 시작 ===")

        from data.collector import DataCollector
        from data.processor import DataProcessor

        collector = DataCollector(self.settings)

        # 최근 5 영업일만 수집 (증분 업데이트)
        end_date   = date.today()
        start_date = end_date - timedelta(days=10)

        logger.info(f"수집 기간: {start_date} ~ {end_date}")

        try:
            raw_df = collector.collect_all(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if raw_df is not None and len(raw_df) > 0:
                # 기존 처리 데이터에 append
                processed_path = Path(
                    self.settings["paths"]["processed_data"]
                ) / "processed_data.parquet"

                processor = DataProcessor()
                new_df = processor.process(raw_df)

                if processed_path.exists():
                    old_df = pd.read_parquet(processed_path)
                    # 중복 제거 후 합치기
                    combined = pd.concat([old_df, new_df]).drop_duplicates(
                        subset=["date", "ticker"]
                    ).sort_values(["date", "ticker"])
                    combined.to_parquet(processed_path, index=False)
                    logger.info(
                        f"데이터 업데이트: {len(old_df):,}행 → {len(combined):,}행 "
                        f"(+{len(combined)-len(old_df):,}행)"
                    )
                else:
                    new_df.to_parquet(processed_path, index=False)
                    logger.info(f"데이터 새로 저장: {len(new_df):,}행")

        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            raise

        logger.info("=== Step 1: 데이터 수집 완료 ===")

    # ------------------------------------------------------------------
    # Step 2: 신호 생성
    # ------------------------------------------------------------------

    def step_signal(self, use_cache: bool = True) -> tuple:
        """모델 추론 → 섹터 가중치 + 종목 랭킹 생성.

        당일 신호가 이미 생성된 경우 캐시 파일에서 로드 (추론 스킵).
        수동으로 --step order를 여러 번 실행해도 추론은 하루 1회만 실행.

        Args:
            use_cache: True면 당일 캐시 사용, False면 강제 재추론

        Returns:
            (final_weights, sector_top_tickers)
        """
        import json
        today_str  = date.today().strftime("%Y%m%d")
        cache_path = Path("tracking") / f"signal_cache_{today_str}.json"

        # 당일 캐시 존재 시 로드
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
                final_weights       = np.array(cache["final_weights"])
                sector_top_tickers  = cache["sector_top_tickers"]
                logger.info(f"=== Step 2: 당일 신호 캐시 로드 ({cache_path.name}) ===")
                return final_weights, sector_top_tickers
            except Exception as e:
                logger.warning(f"캐시 로드 실패 → 재추론: {e}")

        logger.info("=== Step 2: 신호 생성 시작 ===")

        import torch
        import yaml as _yaml
        from pipeline.inference_pipeline import InferencePipeline

        pipeline = InferencePipeline(
            config_path=self.cfg["model"]["config_path"],
            ensemble_path=self.cfg["model"]["ensemble_path"],
        )

        try:
            signals = pipeline.generate_signals()
        except Exception as e:
            logger.error(f"추론 실패: {e}")
            raise

        # 섹터별 신호 추출
        from live.sector_instruments import get_sector_order, SECTOR_ORDER
        sector_order = SECTOR_ORDER

        # inference_pipeline이 반환하는 sector_allocations 사용
        sector_alloc = signals.get("sector_allocations", {})
        n_sectors = len(sector_order)

        raw_weights = np.zeros(n_sectors)
        for i, sector in enumerate(sector_order):
            raw_weights[i] = sector_alloc.get(sector, 0.0)

        # top-K 정규화 (상위 5개 섹터, 양수만)
        top_k = 5
        top_indices = np.argsort(raw_weights)[-top_k:]
        model_weights = np.zeros(n_sectors)
        for idx in top_indices:
            if raw_weights[idx] > 0:
                model_weights[idx] = raw_weights[idx]
        if model_weights.sum() > 1e-8:
            model_weights /= model_weights.sum()
        else:
            model_weights = np.ones(n_sectors) / n_sectors

        # Alpha 블렌딩 (40% model + 60% equal-weight)
        ew = np.ones(n_sectors) / n_sectors
        final_weights = self.alpha_blend * model_weights + (1 - self.alpha_blend) * ew

        # 섹터별 top 종목 추출
        sector_top_tickers = signals.get("sector_top_tickers", {})

        # 신호 기록
        from tracking.trade_log import TradeLogger
        trade_logger = TradeLogger(self.cfg["logging"]["trade_log_db"])
        signals_dict  = {s: float(raw_weights[i]) for i, s in enumerate(sector_order)}
        weights_dict  = {s: float(final_weights[i]) for i, s in enumerate(sector_order)}
        trade_logger.log_signals(signals_dict, weights_dict)

        logger.info("섹터 배분:")
        for i, sector in enumerate(sector_order):
            top = sector_top_tickers.get(sector, [])
            top_str = ", ".join(f"{t['ticker']}({t['score']:+.3f})" for t in top[:3])
            logger.info(
                f"  {sector:<30}: {final_weights[i]:.1%}  [{top_str}]"
            )

        # 당일 캐시 저장 (이후 --step order 재실행 시 추론 스킵)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "date":               today_str,
                    "final_weights":      final_weights.tolist(),
                    "sector_top_tickers": sector_top_tickers,
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"신호 캐시 저장: {cache_path.name}")
        except Exception as e:
            logger.warning(f"신호 캐시 저장 실패: {e}")

        logger.info("=== Step 2: 신호 생성 완료 ===")
        return final_weights, sector_top_tickers

    # ------------------------------------------------------------------
    # Step 2-B: 하락 신호 즉시 매도 (매일 실행)
    # ------------------------------------------------------------------

    def step_sell_check(self, sector_top_tickers: dict) -> list[dict]:
        """매일 06:30 실행: 보유 종목 중 하락 신호 → 즉시 매도."""
        logger.info("=== Step 2-B: 하락 신호 매도 체크 ===")

        from live.signal_to_order import OrderGenerator
        generator = OrderGenerator(config_path="config/live_config.yaml")

        try:
            executed = generator.execute_sell_check(sector_top_tickers)
        except Exception as e:
            logger.error(f"매도 체크 실패: {e}")
            raise

        logger.info(f"=== Step 2-B: 완료 ({len(executed)}건 매도) ===")
        return executed

    # ------------------------------------------------------------------
    # Step 3: 리밸런싱 주문
    # ------------------------------------------------------------------

    def step_order(
        self,
        sector_weights: np.ndarray,
        sector_top_tickers: dict = None,
        twap_wave: int = 1,
    ) -> list[dict]:
        """매일 실행: 신호 기반 리밸런싱 (TWAP 분할 주문).

        Args:
            sector_weights:     섹터별 목표 비중 배열
            sector_top_tickers: 섹터별 top 종목 정보
            twap_wave:          TWAP 파 번호 (1=09:10 40%, 2=11:00 35%, 3=13:30 25%)

        장 휴장일(토/일)에는 스킵.
        """
        logger.info(f"=== Step 3: 리밸런싱 주문 시작 (Wave {twap_wave}) ===")

        today = date.today()
        if today.weekday() >= 5:  # 토=5, 일=6
            if not getattr(self, "_force", False):
                logger.info(f"오늘은 {today.strftime('%A')} → 장 휴장, 스킵")
                return []
            logger.warning(f"오늘은 {today.strftime('%A')} → --force 플래그로 강제 실행")

        from live.signal_to_order import OrderGenerator
        generator = OrderGenerator(
            config_path="config/live_config.yaml",
            twap_wave=twap_wave,
        )

        try:
            executed = generator.execute_rebalance(sector_weights, sector_top_tickers)
        except Exception as e:
            logger.error(f"주문 실행 실패 (Wave {twap_wave}): {e}")
            raise

        logger.info(f"=== Step 3: 리밸런싱 완료 Wave {twap_wave} ({len(executed)}건) ===")
        return executed

    # ------------------------------------------------------------------
    # Step 4: 종가 기록 (End of Day)
    # ------------------------------------------------------------------

    def step_eod(self):
        """종가 기준 성과 업데이트."""
        logger.info("=== Step 4: 종가 기록 시작 ===")

        from broker.kis_api import KISApi
        from tracking.trade_log import TradeLogger

        api = KISApi(mode=self.cfg["broker"]["mode"])
        trade_logger = TradeLogger(self.cfg["logging"]["trade_log_db"])

        try:
            if self.execution_market == "kospi":
                balance = api.get_domestic_balance()
            else:
                balance = api.get_overseas_balance()

            portfolio_value = balance.get("total_eval", 0) + balance.get("cash", 0)

            # 어제 포트폴리오 가치 조회
            perf_df = trade_logger.get_performance_df(days=2)
            if len(perf_df) >= 1:
                prev_value = perf_df["portfolio_value"].iloc[-1]
                daily_return = (portfolio_value - prev_value) / prev_value
            else:
                daily_return = 0.0

            trade_logger.log_daily_performance(
                portfolio_value=portfolio_value,
                daily_return=daily_return,
                n_trades=len(trade_logger.get_trades_df(days=1)),
            )

            logger.info(
                f"포트폴리오: {portfolio_value:,.0f}원, "
                f"일수익률: {daily_return:.2%}"
            )

        except Exception as e:
            logger.error(f"종가 기록 실패: {e}")

        logger.info("=== Step 4: 종가 기록 완료 ===")

    # ------------------------------------------------------------------
    # 전체 파이프라인 실행
    # ------------------------------------------------------------------

    def run_full(self):
        """전체 일간 파이프라인 순서대로 실행."""
        logger.info(f"\n{'='*60}")
        logger.info(f"일간 파이프라인 시작: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        self.step_collect()
        weights, top_tickers = self.step_signal()
        self.step_sell_check(top_tickers)
        self.step_order(weights, top_tickers, twap_wave=1)
        # EOD는 장 마감 후 별도 실행 (step_eod)

        elapsed = time.time() - start_time
        logger.info(f"일간 파이프라인 완료: {elapsed:.0f}초 소요")

    # ------------------------------------------------------------------
    # 스케줄러 데몬
    # ------------------------------------------------------------------

    def _daemon_signal_and_sell(self):
        """데몬용: 신호 생성 + 하락 신호 즉시 매도."""
        weights, top_tickers = self.step_signal()
        self._last_weights = weights
        self._last_top_tickers = top_tickers
        self.step_sell_check(top_tickers)

    def _daemon_order(self, twap_wave: int = 1):
        """데몬용: 저장된 신호로 TWAP 파별 리밸런싱 주문."""
        weights    = getattr(self, "_last_weights",     np.ones(11) / 11)
        top_tickers = getattr(self, "_last_top_tickers", {})
        self.step_order(weights, top_tickers, twap_wave=twap_wave)

    def run_daemon(self):
        """스케줄러 데몬: 지정 시간에 자동 실행 (TWAP 3-파)."""
        import schedule

        schedule_cfg = self.cfg["schedule"]
        order_times  = schedule_cfg["order_waves"]          # wave1/wave2/wave3

        schedule.every().day.at(schedule_cfg["data_collect_time"]).do(self.step_collect)
        schedule.every().day.at(schedule_cfg["signal_gen_time"]).do(self._daemon_signal_and_sell)

        # TWAP 3-파 주문 스케줄
        schedule.every().day.at(order_times["wave1"]).do(
            lambda: self._daemon_order(twap_wave=1)
        )
        schedule.every().day.at(order_times["wave2"]).do(
            lambda: self._daemon_order(twap_wave=2)
        )
        schedule.every().day.at(order_times["wave3"]).do(
            lambda: self._daemon_order(twap_wave=3)
        )

        schedule.every().day.at(schedule_cfg["eod_record_time"]).do(self.step_eod)

        logger.info("스케줄러 데몬 시작 (TWAP 3-파):")
        logger.info(f"  데이터 수집:  {schedule_cfg['data_collect_time']}")
        logger.info(f"  신호 생성:    {schedule_cfg['signal_gen_time']}")
        logger.info(f"  주문 Wave 1:  {order_times['wave1']} (40%)")
        logger.info(f"  주문 Wave 2:  {order_times['wave2']} (35%)")
        logger.info(f"  주문 Wave 3:  {order_times['wave3']} (25%)")
        logger.info(f"  종가 기록:    {schedule_cfg['eod_record_time']}")

        while True:
            schedule.run_pending()
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="일간 자동 실행 스케줄러")
    parser.add_argument("--step", choices=["collect", "signal", "order", "eod", "all"],
                        default="all", help="실행할 단계")
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], default=1,
                        help="TWAP 파 번호 (--step order 와 함께 사용)")
    parser.add_argument("--daemon", action="store_true", help="스케줄러 데몬 실행")
    parser.add_argument("--force", action="store_true", help="주말/휴장 체크 무시 (테스트용)")
    parser.add_argument("--config", default="config/live_config.yaml")
    args = parser.parse_args()

    runner = DailyRunner(config_path=args.config)
    runner._force = args.force

    if args.daemon:
        runner.run_daemon()
    elif args.step == "collect":
        runner.step_collect()
    elif args.step == "signal":
        weights, top_tickers = runner.step_signal()
        print("\n섹터 가중치:", weights)
    elif args.step == "order":
        weights, top_tickers = runner.step_signal()
        runner.step_order(weights, top_tickers, twap_wave=args.wave)
    elif args.step == "eod":
        runner.step_eod()
    else:
        runner.run_full()


if __name__ == "__main__":
    main()
