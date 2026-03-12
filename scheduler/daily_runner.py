"""일간 자동 실행 스케줄러.

매일 실행 순서:
  06:00 - 데이터 수집 (yfinance + pykrx, NASDAQ 전일 종가 포함)
  06:10 - [US] NASDAQ 종가 기반 신호 생성 + 저장 (하락 신호 즉시 매도)
  06:30 - [KR] 신호 생성 + 하락 신호 즉시 매도
  09:10 - [KR] TWAP Wave 1 (40%) — 장 시작 10분 후
  11:00 - [KR] TWAP Wave 2 (35%) — 오전 중반
  13:30 - [KR] TWAP Wave 3 (25%) — 오후 장
  16:00 - 종가 기록 + 성과 업데이트
  23:40 - [US] TWAP Wave 1 (40%) — 미국 장 시작 10분 후
  02:00 - [US] TWAP Wave 2 (35%) — 미국 오전 중반
  04:30 - [US] TWAP Wave 3 (25%) — 미국 오후 장

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
from utils.ticker_utils import is_domestic


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
        self._kr_layer2_scales: dict = {}   # Layer 2 장중 신호 (KR)
        self._us_layer2_scales: dict = {}   # Layer 2 장중 신호 (US)

        from strategy.signal import MarketRegimeDetector
        self._regime_detector = MarketRegimeDetector()
        self._current_regime: dict = {"regime": "neutral", "scale_factor": 1.0, "alpha": 0.4}

        # 헤지 매니저
        from strategy.hedge import PortfolioHedger
        self._hedger = PortfolioHedger()

        # DART 공시 이벤트 클라이언트
        try:
            from data.dart_client import DartClient
            self._dart_client = DartClient()
            self._dart_available = self._dart_client.is_available
        except ImportError:
            self._dart_available = False
            logger.info("DART 클라이언트 미설치 — 공시 이벤트 비활성화")

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

                # 증분 수집이므로 min_history_days=1 (히스토리 필터 비활성화)
                # 기존 parquet과 merge 후 drop_duplicates로 정리됨
                processor = DataProcessor(min_history_days=1)
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
            return  # 수집 실패해도 데몬 유지 — step_signal이 기존 parquet 재사용

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

    def step_sell_check(
        self,
        sector_top_tickers: dict,
        market_filter: str = None,
    ) -> list[dict]:
        """보유 종목 중 하락 신호 → 즉시 매도.

        Args:
            sector_top_tickers: 섹터별 top 종목
            market_filter: None=전체, "domestic"=국내만, "overseas"=해외만
        """
        market_label = f"[{market_filter}] " if market_filter else ""
        logger.info(f"=== Step 2-B: {market_label}하락 신호 매도 체크 ===")

        from live.signal_to_order import OrderGenerator
        generator = OrderGenerator(config_path="config/live_config.yaml")

        try:
            executed = generator.execute_sell_check(sector_top_tickers, market_filter=market_filter)
        except Exception as e:
            logger.error(f"매도 체크 실패: {e}")
            raise

        logger.info(f"=== Step 2-B: {market_label}완료 ({len(executed)}건 매도) ===")
        return executed

    # ------------------------------------------------------------------
    # 시장별 신호 분리 헬퍼
    # ------------------------------------------------------------------

    def _filter_by_market(self, sector_top_tickers: dict, market: str) -> dict:
        """sector_top_tickers에서 특정 시장 종목만 필터링.

        market: "domestic" → .KS/.KQ 종목만 (KOSPI/KOSDAQ)
                "overseas" → 그 외 종목 (NASDAQ/S&P500)
        빈 섹터는 제외.
        """
        filtered = {}
        for sector, tickers in sector_top_tickers.items():
            if market == "domestic":
                kept = [t for t in tickers if is_domestic(t["ticker"])]
            else:
                kept = [t for t in tickers if not is_domestic(t["ticker"])]
            if kept:
                filtered[sector] = kept
        return filtered

    def _save_market_cache(self, weights, top_tickers: dict, market: str):
        """시장별 신호 캐시 저장 (signal_cache_{market}_{today}.json)."""
        today_str  = date.today().strftime("%Y%m%d")
        cache_path = Path("tracking") / f"signal_cache_{market}_{today_str}.json"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "date":               today_str,
                    "market":             market,
                    "final_weights":      weights.tolist(),
                    "sector_top_tickers": top_tickers,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"[{market.upper()}] 시장별 신호 캐시 저장: {cache_path.name}")
        except Exception as e:
            logger.warning(f"[{market.upper()}] 캐시 저장 실패: {e}")

    def _load_market_cache(self, market: str) -> tuple:
        """시장별 신호 캐시 로드. 없으면 전체 캐시에서 필터링.

        market: "kr" (domestic) 또는 "us" (overseas)
        """
        today_str  = date.today().strftime("%Y%m%d")
        cache_path = Path("tracking") / f"signal_cache_{market}_{today_str}.json"

        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
                weights     = np.array(cache["final_weights"])
                top_tickers = cache["sector_top_tickers"]
                logger.info(f"[{market.upper()}] 시장별 캐시 로드: {cache_path.name}")
                return weights, top_tickers
            except Exception as e:
                logger.warning(f"[{market.upper()}] 시장별 캐시 로드 실패 → 전체 캐시 폴백: {e}")

        # 폴백: 전체 캐시 로드 후 시장 필터 적용
        logger.info(f"[{market.upper()}] 시장별 캐시 없음 → 전체 캐시 필터링")
        weights, top_tickers = self.step_signal(use_cache=True)
        market_key   = "domestic" if market == "kr" else "overseas"
        top_tickers  = self._filter_by_market(top_tickers, market_key)
        return weights, top_tickers

    # ------------------------------------------------------------------
    # Step 3: 리밸런싱 주문
    # ------------------------------------------------------------------

    def step_order(
        self,
        sector_weights: np.ndarray,
        sector_top_tickers: dict = None,
        twap_wave: int = 1,
        market_filter: str = None,
        layer2_scales: dict = None,
    ) -> list[dict]:
        """매일 실행: 신호 기반 리밸런싱 (TWAP 분할 주문).

        Args:
            sector_weights:     섹터별 목표 비중 배열
            sector_top_tickers: 섹터별 top 종목 정보
            twap_wave:          TWAP 파 번호
            market_filter:      None=전체, "domestic"=국내만, "overseas"=해외만
            layer2_scales:      Layer 2 장중 신호 스케일

        장 휴장일(토/일)에는 스킵.
        """
        market_label = f"[{market_filter}] " if market_filter else ""
        logger.info(f"=== Step 3: {market_label}리밸런싱 주문 시작 (Wave {twap_wave}) ===")

        today = date.today()
        if today.weekday() >= 5:
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
            executed = generator.execute_rebalance(
                sector_weights, sector_top_tickers,
                market_filter=market_filter,
                layer2_scales=layer2_scales,
            )
        except Exception as e:
            logger.error(f"주문 실행 실패 ({market_label}Wave {twap_wave}): {e}")
            return []

        logger.info(f"=== Step 3: {market_label}리밸런싱 완료 Wave {twap_wave} ({len(executed)}건) ===")
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
            paper_trading = self.cfg["broker"].get("paper_trading", False)

            # split 모드 또는 paper_trading: 국내 잔고 기준
            # (해외 sandbox USD $0 → overseas balance $0 → NaN return 방지)
            if self.execution_market in ("kospi", "split") or paper_trading:
                balance = api.get_domestic_balance()
            else:
                balance = api.get_overseas_balance()

            # KIS total_eval = 주식+현금 합산값, cash 따로 더하면 이중 계산
            portfolio_value = balance.get("total_eval", 0)

            # 어제 포트폴리오 가치 조회
            perf_df = trade_logger.get_performance_df(days=2)
            if len(perf_df) >= 1:
                prev_value = float(perf_df["portfolio_value"].iloc[-1])
                if prev_value > 0:
                    daily_return = (portfolio_value - prev_value) / prev_value
                else:
                    daily_return = 0.0
            else:
                daily_return = 0.0

            # NaN/Inf 방어 (NOT NULL constraint 위반 방지)
            import math
            if math.isnan(daily_return) or math.isinf(daily_return):
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
        """데몬용: [KR] 06:30 — 전체 신호에서 국내 종목 분리 + 하락 신호 즉시 매도.

        06:10에 _daemon_us_signal()이 이미 전체 추론을 마치고 캐시를 저장함.
        여기서는 재추론 없이 캐시 로드 → 국내 종목만 필터 → KR 전용 캐시 저장.
        이렇게 해야 KOSPI 신호가 NASDAQ 신호(06:10 기준)와 분리됨.
        """
        logger.info("=== [KR] KOSPI 신호 분리 및 하락 매도 ===")
        weights, top_tickers = self.step_signal(use_cache=True)  # 06:10 캐시 재사용
        kr_top_tickers       = self._filter_by_market(top_tickers, "domestic")
        self._last_weights     = weights
        self._last_top_tickers = kr_top_tickers
        # KR 전용 캐시 저장 (Wave 1/2/3 폴백용)
        self._save_market_cache(weights, kr_top_tickers, "kr")
        # 국내 포지션만 하락 신호 체크
        self.step_sell_check(kr_top_tickers, market_filter="domestic")
        logger.info(
            f"[KR] 분리 완료: {sum(len(v) for v in kr_top_tickers.values())}개 국내 종목"
        )

    def _daemon_kr_order(self, twap_wave: int = 1):
        """데몬용: 국내(KR) 종목만 TWAP 파별 주문."""
        weights     = getattr(self, "_last_weights",     None)
        top_tickers = getattr(self, "_last_top_tickers", None)
        if weights is None or not top_tickers:
            logger.info("[KR] 메모리 신호 없음 → KR 전용 캐시에서 로드")
            weights, top_tickers = self._load_market_cache("kr")
        layer2 = self._kr_layer2_scales
        self.step_order(weights, top_tickers, twap_wave=twap_wave, market_filter="domestic", layer2_scales=layer2)

    def _daemon_us_signal(self):
        """데몬용: NASDAQ 종가 직후 (06:10) 전체 추론 + 해외 신호 분리 저장.

        06:00 수집 직후 실행. 캐시 무시(use_cache=False)로 신선한 신호 생성.
        전체 캐시(domestic+overseas) 저장 후 해외만 필터한 US 전용 캐시도 저장.
        06:30 _daemon_signal_and_sell은 전체 캐시를 재사용하므로 재추론 불필요.
        """
        logger.info("=== [US] NASDAQ 종가 기반 전체 추론 시작 ===")
        try:
            weights, top_tickers = self.step_signal(use_cache=False)  # 전체 추론
            us_top_tickers       = self._filter_by_market(top_tickers, "overseas")
            self._us_weights     = weights
            self._us_top_tickers = us_top_tickers
            # US 전용 캐시 저장 (Wave 1/2/3 폴백용)
            self._save_market_cache(weights, us_top_tickers, "us")
            # 해외 포지션만 하락 신호 체크
            self.step_sell_check(us_top_tickers, market_filter="overseas")
            logger.info(
                f"[US] 신호 저장 완료: {sum(len(v) for v in us_top_tickers.values())}개 해외 종목"
                " → 23:40/02:00/04:30 집행 예정"
            )
        except Exception as e:
            logger.error(f"[US] 신호 생성 실패: {e}")

        # 레짐 감지 (시장 레짐 업데이트) + 헤지 신호
        perf_df = None
        try:
            from tracking.trade_log import TradeLogger
            trade_log = TradeLogger(self.cfg["logging"]["trade_log_db"])
            perf_df = trade_log.get_performance_df(days=90)
            if len(perf_df) >= 20 and "daily_return" in perf_df.columns:
                market_rets = pd.Series(
                    perf_df["daily_return"].values,
                    index=pd.to_datetime(perf_df["date"])
                )
                self._current_regime = self._regime_detector.detect(market_rets)
                logger.info(
                    f"시장 레짐: {self._current_regime['regime']} "
                    f"(scale={self._current_regime['scale_factor']:.1f}, "
                    f"alpha={self._current_regime['alpha']:.2f}) — "
                    f"{self._current_regime['details']}"
                )
        except Exception as e:
            logger.warning(f"레짐 감지 실패: {e}")

        # 헤지 신호 계산
        try:
            if perf_df is not None and len(perf_df) >= 20 and "daily_return" in perf_df.columns:
                port_rets = pd.Series(perf_df["daily_return"].values, index=pd.to_datetime(perf_df["date"]))
                hedge_signal = self._hedger.compute_hedge_signal(
                    port_rets,
                    regime=self._current_regime.get("regime", "neutral"),
                )
                if hedge_signal["should_rebalance"]:
                    logger.info(
                        f"헤지 신호: ratio={hedge_signal['hedge_ratio']:.0%}, "
                        f"beta={hedge_signal['beta']:.2f} — {hedge_signal['reason']}"
                    )
                    for inst, ratio in hedge_signal["instruments"].items():
                        if ratio > 0:
                            logger.info(f"  {inst}: {ratio:.1%}")
        except Exception as e:
            logger.warning(f"헤지 신호 계산 실패: {e}")

    def _daemon_us_order(self, twap_wave: int = 1):
        """데몬용: 해외(US) 종목 TWAP 파별 주문 (06:10 생성된 신호 사용)."""
        weights     = getattr(self, "_us_weights",     None)
        top_tickers = getattr(self, "_us_top_tickers", None)
        if weights is None or not top_tickers:
            logger.info("[US] 메모리 신호 없음 → US 전용 캐시에서 로드")
            weights, top_tickers = self._load_market_cache("us")
        layer2 = self._us_layer2_scales
        self.step_order(weights, top_tickers, twap_wave=twap_wave, market_filter="overseas", layer2_scales=layer2)

    # ------------------------------------------------------------------
    # 신호 경과 시간 계산
    # ------------------------------------------------------------------

    def _get_signal_age_hours(self, base_time_str: str = "06:10") -> float:
        """신호 생성 시각부터 현재까지 경과 시간 (hours)."""
        from datetime import datetime as _dt
        now = _dt.now()
        h, m = map(int, base_time_str.split(":"))
        base = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now < base:
            base -= timedelta(days=1)
        return (now - base).total_seconds() / 3600.0

    # ------------------------------------------------------------------
    # Layer 2: 장중 인트라데이 신호 (KIS/yfinance 현재가, 추론 없음)
    # ------------------------------------------------------------------

    def _compute_layer2_signal(self, sector_top_tickers: dict, market: str = "domestic") -> dict:
        """Layer 2 장중 신호: KIS API 현재가 기반 인트라데이 모멘텀 계산 (수초).

        Args:
            market: "domestic" (KIS API) or "overseas" (yfinance)

        Returns:
            {ticker_normalized: scale_factor (0.0~1.5)}
            1.0=중립, >1.0=모멘텀 강화, <1.0=약화
        """
        from broker.kis_api import KISApi
        from utils.ticker_utils import kis_code, is_domestic

        scales: dict = {}

        if market == "domestic":
            try:
                api = KISApi(mode=self.cfg["broker"]["mode"], market_type="domestic")
            except Exception as e:
                logger.warning(f"Layer2 KIS API 초기화 실패: {e}")
                return scales

            for sector, tickers in sector_top_tickers.items():
                for tkr_info in tickers:
                    ticker = tkr_info["ticker"]
                    if not is_domestic(ticker):
                        continue
                    norm = kis_code(ticker)
                    try:
                        d = api.get_domestic_price(norm)
                        current    = float(d["price"])
                        open_p     = float(d.get("open", current))
                        change_pct = float(d.get("change_pct", 0)) / 100.0
                        prev_close = current / (1 + change_pct) if abs(change_pct) > 1e-6 else current

                        gap           = (open_p - prev_close) / prev_close if prev_close > 0 else 0
                        intraday_move = (current - open_p) / open_p if open_p > 0 else 0

                        # 갭 연속성
                        if abs(gap) > 0.005:
                            gap_cont = 1.2 if gap * intraday_move > 0 else 0.7
                        else:
                            gap_cont = 1.0

                        # 장중 모멘텀 스케일
                        if   intraday_move >  0.03: mom = 1.5
                        elif intraday_move >  0.02: mom = 1.3
                        elif intraday_move >  0.01: mom = 1.1
                        elif intraday_move < -0.04: mom = 0.1  # 장중 -4% → 긴급
                        elif intraday_move < -0.03: mom = 0.3
                        elif intraday_move < -0.02: mom = 0.6
                        elif intraday_move < -0.01: mom = 0.8
                        else:                       mom = 1.0

                        scales[norm] = round(min(gap_cont * mom, 1.5), 3)
                    except Exception as e:
                        logger.debug(f"Layer2 KR [{ticker}]: {e}")
                        scales[norm] = 1.0

        else:  # overseas — yfinance 5분봉 사용
            import yfinance as yf
            from utils.ticker_utils import kis_code, is_domestic
            for sector, tickers in sector_top_tickers.items():
                for tkr_info in tickers:
                    ticker = tkr_info["ticker"]
                    if is_domestic(ticker):
                        continue
                    try:
                        df = yf.Ticker(ticker).history(period="1d", interval="5m")
                        if df.empty:
                            scales[ticker] = 1.0
                            continue
                        open_p  = float(df["Open"].iloc[0])
                        current = float(df["Close"].iloc[-1])
                        intraday_move = (current - open_p) / open_p if open_p > 0 else 0

                        if   intraday_move >  0.03: mom = 1.5
                        elif intraday_move >  0.02: mom = 1.3
                        elif intraday_move >  0.01: mom = 1.1
                        elif intraday_move < -0.04: mom = 0.1
                        elif intraday_move < -0.03: mom = 0.3
                        elif intraday_move < -0.02: mom = 0.6
                        elif intraday_move < -0.01: mom = 0.8
                        else:                       mom = 1.0

                        scales[ticker] = round(min(mom, 1.5), 3)
                    except Exception as e:
                        logger.debug(f"Layer2 US [{ticker}]: {e}")
                        scales[ticker] = 1.0

        logger.info(f"Layer 2 [{market}] 계산 완료: {len(scales)}개 종목")
        return scales

    # ------------------------------------------------------------------
    # KR Layer 2 장중 업데이트
    # ------------------------------------------------------------------

    def _daemon_kr_intraday_update(self, update_no: int = 1):
        """데몬용: [KR] 장중 Layer 2 신호 업데이트 (10:00, 13:00).

        KIS API로 현재가 조회 → 인트라데이 모멘텀 스케일 계산 → 메모리 저장.
        다음 Wave 실행 시 이 스케일이 주문 수량에 반영됨.
        """
        # 스탑로스 체크
        try:
            self._check_intraday_stoploss(stoploss_pct=0.03, market_filter="domestic")
        except Exception as e:
            logger.error(f"스탑로스 체크 오류: {e}")

        logger.info(f"=== [KR] Layer 2 장중 업데이트 #{update_no} ===")
        top_tickers = getattr(self, "_last_top_tickers", None)
        if not top_tickers:
            _, top_tickers = self._load_market_cache("kr")
        if not top_tickers:
            logger.info("[KR] Layer 2: 신호 없음 → 스킵")
            return
        try:
            scales = self._compute_layer2_signal(top_tickers, market="domestic")
            self._kr_layer2_scales = scales
            strong = sum(1 for v in scales.values() if v >= 1.1)
            weak   = sum(1 for v in scales.values() if v <= 0.7)
            logger.info(
                f"[KR] Layer 2 #{update_no}: 총 {len(scales)}개 "
                f"(강화 {strong}개 ≥1.1 / 약화 {weak}개 ≤0.7)"
            )
        except Exception as e:
            logger.error(f"[KR] Layer 2 업데이트 실패: {e}")

        # DART 공시 이벤트 확인
        try:
            dart_scales = self._check_dart_events(market_filter="domestic")
            if dart_scales:
                # 기존 Layer 2 스케일에 DART 이벤트 스케일 병합
                for ticker, scale in dart_scales.items():
                    self._kr_layer2_scales[ticker] = self._kr_layer2_scales.get(ticker, 1.0) * scale
                logger.info(f"DART 이벤트 → Layer 2 스케일 반영: {len(dart_scales)}종목")
        except Exception as e:
            logger.warning(f"DART 이벤트 처리 오류: {e}")

    def _check_dart_events(self, market_filter: str = "domestic") -> dict:
        """DART 공시 이벤트 확인 및 Layer 2 스케일에 반영.

        강한 부정적 공시 → Layer 2 스케일 0.3 (축소)
        강한 긍정적 공시 → Layer 2 스케일 1.3 (확대)

        Returns:
            {ticker_normalized: scale_factor} 이벤트 기반 스케일
        """
        if not self._dart_available:
            return {}

        try:
            event_signals = self._dart_client.get_event_signals(days_back=1)
            if not event_signals:
                return {}

            scales = {}
            for corp_name, info in event_signals.items():
                score = info["score"]
                if score < -0.5:
                    scale = 0.3  # 강한 부정적 → 대폭 축소
                elif score < -0.2:
                    scale = 0.6  # 약한 부정적 → 축소
                elif score > 0.5:
                    scale = 1.3  # 강한 긍정적 → 확대
                elif score > 0.2:
                    scale = 1.15  # 약한 긍정적 → 소폭 확대
                else:
                    continue

                scales[corp_name] = scale
                logger.info(
                    f"[DART 이벤트] {corp_name}: score={score:+.2f} → scale={scale:.1f} "
                    f"({info['events'][0][:50]})"
                )

            return scales

        except Exception as e:
            logger.warning(f"DART 이벤트 확인 실패: {e}")
            return {}

    # ------------------------------------------------------------------
    # US Layer 2 장중 업데이트
    # ------------------------------------------------------------------

    def _daemon_us_intraday_update(self, update_no: int = 1):
        """데몬용: [US] 장중 Layer 2 신호 업데이트 (01:00, 03:30 KST)."""
        # 스탑로스 체크
        try:
            self._check_intraday_stoploss(stoploss_pct=0.03, market_filter="overseas")
        except Exception as e:
            logger.error(f"US 스탑로스 체크 오류: {e}")

        logger.info(f"=== [US] Layer 2 장중 업데이트 #{update_no} ===")
        top_tickers = getattr(self, "_us_top_tickers", None)
        if not top_tickers:
            _, top_tickers = self._load_market_cache("us")
        if not top_tickers:
            return
        try:
            scales = self._compute_layer2_signal(top_tickers, market="overseas")
            self._us_layer2_scales = scales
            strong = sum(1 for v in scales.values() if v >= 1.1)
            weak   = sum(1 for v in scales.values() if v <= 0.7)
            logger.info(
                f"[US] Layer 2 #{update_no}: 총 {len(scales)}개 "
                f"(강화 {strong}개 / 약화 {weak}개)"
            )
        except Exception as e:
            logger.error(f"[US] Layer 2 업데이트 실패: {e}")

    # ------------------------------------------------------------------
    # 인트라데이 스탑로스 체크
    # ------------------------------------------------------------------

    def _check_intraday_stoploss(
        self,
        stoploss_pct: float = 0.03,
        market_filter: str = None,
    ) -> list:
        """인트라데이 스탑로스 체크.

        각 포지션의 현재 손실이 stoploss_pct 초과 시 즉시 매도.

        Args:
            stoploss_pct: 스탑로스 임계값 (기본 3%)
            market_filter: "domestic", "overseas", or None

        Returns:
            List of triggered stop-loss sell results
        """
        logger.info(f"[스탑로스 체크] 임계값={stoploss_pct:.0%}")
        from live.signal_to_order import OrderGenerator
        gen = OrderGenerator(config_path="config/live_config.yaml")

        try:
            current_positions = gen._get_current_positions()
        except Exception as e:
            logger.warning(f"포지션 조회 실패: {e}")
            return []

        stop_orders = []
        from utils.ticker_utils import is_domestic, kis_code
        import yfinance as yf

        for ticker, pos in current_positions.items():
            qty = pos.get("qty", 0)
            if qty <= 0:
                continue

            # 시장 필터
            is_dom = is_domestic(ticker)
            if market_filter == "domestic" and not is_dom:
                continue
            if market_filter == "overseas" and is_dom:
                continue

            # 평균단가 조회
            avg_price = pos.get("avg_price", 0)
            if avg_price <= 0:
                continue

            # 현재가 조회
            try:
                if is_dom:
                    api = gen.api_domestic
                    current_price = float(api.get_domestic_price(kis_code(ticker))["price"])
                else:
                    try:
                        api = gen.api_overseas
                        current_price = float(api.get_overseas_price(ticker, "NAS")["price"])
                    except Exception:
                        current_price = float(yf.Ticker(ticker).fast_info.last_price)
            except Exception as e:
                logger.debug(f"{ticker} 현재가 조회 실패: {e}")
                continue

            loss_pct = (current_price - avg_price) / avg_price

            if loss_pct < -stoploss_pct:
                logger.warning(
                    f"[스탑로스 발동] {ticker}: avg={avg_price:.0f}, "
                    f"current={current_price:.0f}, loss={loss_pct:.1%}"
                )
                stop_orders.append({
                    "ticker":        ticker,
                    "side":          "sell",
                    "qty":           qty,
                    "sector":        pos.get("sector", "unknown"),
                    "market":        "domestic" if is_dom else "overseas",
                    "exchange":      "",
                    "score":         0.0,
                    "target_amount": 0,
                })

        if not stop_orders:
            logger.info("[스탑로스 체크] 발동 없음")
            return []

        logger.warning(f"[스탑로스] {len(stop_orders)}건 발동")
        executed = []
        for order in stop_orders:
            try:
                result = gen._execute_with_retry(order, "aggressive")
                gen.logger.log_trade(result, note="stoploss")
                executed.append(result)
            except Exception as e:
                logger.error(f"스탑로스 실패 [{order['ticker']}]: {e}")

        return executed

    # ------------------------------------------------------------------
    # KR 시간외 세션
    # ------------------------------------------------------------------

    def _daemon_kr_premarket(self):
        """데몬용: [KR] 장전 시간외 단일가 (07:30) — 15% 선진입."""
        logger.info("=== [KR] 장전 시간외 단일가 (07:30) ===")
        weights, top_tickers = self._load_market_cache("kr")
        if not top_tickers:
            logger.info("[KR] 장전: 신호 없음 → 스킵")
            return
        from live.signal_to_order import OrderGenerator
        generator   = OrderGenerator(config_path="config/live_config.yaml")
        signal_age  = self._get_signal_age_hours("06:10")
        try:
            executed = generator.execute_extended_hours(
                sector_weights=weights,
                sector_top_tickers=top_tickers,
                session="premarket_kr",
                signal_age_hours=signal_age,
                market_filter="domestic",
            )
            logger.info(f"[KR] 장전 완료: {len(executed)}건")
        except Exception as e:
            logger.error(f"[KR] 장전 시간외 실패: {e}")

    def _daemon_kr_afterclose(self):
        """데몬용: [KR] 장후 시간외 종가 (15:35) — 당일 모멘텀 강한 종목 5%."""
        logger.info("=== [KR] 장후 시간외 종가 (15:35) ===")
        weights, top_tickers = self._load_market_cache("kr")
        if not top_tickers:
            return
        from live.signal_to_order import OrderGenerator
        generator  = OrderGenerator(config_path="config/live_config.yaml")
        signal_age = self._get_signal_age_hours("06:10")
        layer2     = self._kr_layer2_scales
        try:
            executed = generator.execute_extended_hours(
                sector_weights=weights,
                sector_top_tickers=top_tickers,
                session="afterclose_kr",
                signal_age_hours=signal_age,
                layer2_scales=layer2,
                market_filter="domestic",
            )
            logger.info(f"[KR] 장후 종가 완료: {len(executed)}건")
        except Exception as e:
            logger.error(f"[KR] 장후 종가 실패: {e}")

    def _daemon_kr_aftersingle(self):
        """데몬용: [KR] 장후 시간외 단일가 (16:30) — 다음날 선포지션 5%."""
        logger.info("=== [KR] 장후 시간외 단일가 (16:30) ===")
        weights, top_tickers = self._load_market_cache("kr")
        if not top_tickers:
            return
        from live.signal_to_order import OrderGenerator
        generator  = OrderGenerator(config_path="config/live_config.yaml")
        signal_age = self._get_signal_age_hours("06:10")
        layer2     = self._kr_layer2_scales
        try:
            executed = generator.execute_extended_hours(
                sector_weights=weights,
                sector_top_tickers=top_tickers,
                session="aftersingle_kr",
                signal_age_hours=signal_age,
                layer2_scales=layer2,
                market_filter="domestic",
            )
            logger.info(f"[KR] 장후 단일가 완료: {len(executed)}건")
        except Exception as e:
            logger.error(f"[KR] 장후 단일가 실패: {e}")

    # ------------------------------------------------------------------
    # US 시간외 세션
    # ------------------------------------------------------------------

    def _daemon_us_premarket_order(self):
        """데몬용: [US] Pre-market (18:30 KST) — 15% 선진입."""
        logger.info("=== [US] Pre-market 주문 (18:30) ===")
        weights, top_tickers = self._load_market_cache("us")
        if not top_tickers:
            return
        from live.signal_to_order import OrderGenerator
        generator  = OrderGenerator(config_path="config/live_config.yaml")
        signal_age = self._get_signal_age_hours("06:10")
        try:
            executed = generator.execute_extended_hours(
                sector_weights=weights,
                sector_top_tickers=top_tickers,
                session="premarket_us",
                signal_age_hours=signal_age,
                market_filter="overseas",
            )
            logger.info(f"[US] Pre-market 완료: {len(executed)}건")
        except Exception as e:
            logger.error(f"[US] Pre-market 실패: {e}")

    def _daemon_us_afterhours_order(self):
        """데몬용: [US] After-hours (05:10 KST) — 10% 추가 (장 마감 후 이벤트 반응)."""
        logger.info("=== [US] After-hours 주문 (05:10) ===")
        weights, top_tickers = self._load_market_cache("us")
        if not top_tickers:
            return
        from live.signal_to_order import OrderGenerator
        generator  = OrderGenerator(config_path="config/live_config.yaml")
        # After-hours는 전날 06:10 신호 기준 → 약 23시간 경과
        signal_age = self._get_signal_age_hours("06:10")
        layer2     = self._us_layer2_scales
        try:
            executed = generator.execute_extended_hours(
                sector_weights=weights,
                sector_top_tickers=top_tickers,
                session="afterhours_us",
                signal_age_hours=signal_age,
                layer2_scales=layer2,
                market_filter="overseas",
            )
            logger.info(f"[US] After-hours 완료: {len(executed)}건")
        except Exception as e:
            logger.error(f"[US] After-hours 실패: {e}")

    def run_daemon(self):
        """스케줄러 데몬: NASDAQ 종가 직후 신호 생성 + KR/US TWAP 3-파."""
        import schedule
        try:
            from scheduler import lotto_runner as _lotto_runner
            _lotto_available = True
        except ImportError:
            logger.warning("lotto_runner 모듈 없음 → 로또 분석 스케줄 비활성화")
            _lotto_available = False

        sc = self.cfg["schedule"]
        kr = sc["kr_order_waves"]
        us = sc["us_order_waves"]

        # ── 데이터 수집 ────────────────────────────────────────────────
        schedule.every().day.at(sc["data_collect_time"]).do(self.step_collect)

        # ── [US] After-hours (05:10, 전날 신호 기반) ──────────────────
        schedule.every().day.at(sc.get("us_afterhours_time", "05:10")).do(
            self._daemon_us_afterhours_order
        )

        # ── [US] 신호 생성 (06:10) ────────────────────────────────────
        schedule.every().day.at(sc["us_signal_time"]).do(self._daemon_us_signal)

        # ── [KR] 신호 분리 + 하락 매도 (06:30) ───────────────────────
        schedule.every().day.at(sc["signal_gen_time"]).do(self._daemon_signal_and_sell)

        # ── [KR] 장전 시간외 단일가 (07:30) ──────────────────────────
        schedule.every().day.at(sc.get("kr_premarket_time", "07:30")).do(
            self._daemon_kr_premarket
        )

        # ── [KR] 정규장 Wave 1 (09:10) ────────────────────────────────
        schedule.every().day.at(kr["wave1"]).do(
            lambda: self._daemon_kr_order(twap_wave=1)
        )

        # ── [KR] Layer 2 업데이트 1차 (10:00) ────────────────────────
        schedule.every().day.at(sc.get("kr_intraday_update_1", "10:00")).do(
            lambda: self._daemon_kr_intraday_update(1)
        )

        # ── [KR] 정규장 Wave 2 (11:00) ────────────────────────────────
        schedule.every().day.at(kr["wave2"]).do(
            lambda: self._daemon_kr_order(twap_wave=2)
        )

        # ── [KR] Layer 2 업데이트 2차 (13:00) ────────────────────────
        schedule.every().day.at(sc.get("kr_intraday_update_2", "13:00")).do(
            lambda: self._daemon_kr_intraday_update(2)
        )

        # ── [KR] 정규장 Wave 3 (13:30) ────────────────────────────────
        schedule.every().day.at(kr["wave3"]).do(
            lambda: self._daemon_kr_order(twap_wave=3)
        )

        # ── [KR] 장후 시간외 종가 (15:35) ────────────────────────────
        schedule.every().day.at(sc.get("kr_afterclose_time", "15:35")).do(
            self._daemon_kr_afterclose
        )

        # ── EOD 기록 (16:00) ──────────────────────────────────────────
        schedule.every().day.at(sc["eod_record_time"]).do(self.step_eod)

        # ── [KR] 장후 시간외 단일가 (16:30) ──────────────────────────
        schedule.every().day.at(sc.get("kr_aftersingle_time", "16:30")).do(
            self._daemon_kr_aftersingle
        )

        # ── [US] Pre-market (18:30) ───────────────────────────────────
        schedule.every().day.at(sc.get("us_premarket_time", "18:30")).do(
            self._daemon_us_premarket_order
        )

        # ── [US] 정규장 Wave 1 (23:40) ────────────────────────────────
        schedule.every().day.at(us["wave1"]).do(
            lambda: self._daemon_us_order(twap_wave=1)
        )

        # ── [US] Layer 2 업데이트 1차 (01:00) ────────────────────────
        schedule.every().day.at(sc.get("us_intraday_update_1", "01:00")).do(
            lambda: self._daemon_us_intraday_update(1)
        )

        # ── [US] 정규장 Wave 2 (02:00) ────────────────────────────────
        schedule.every().day.at(us["wave2"]).do(
            lambda: self._daemon_us_order(twap_wave=2)
        )

        # ── [US] Layer 2 업데이트 2차 (03:30) ────────────────────────
        schedule.every().day.at(sc.get("us_intraday_update_2", "03:30")).do(
            lambda: self._daemon_us_intraday_update(2)
        )

        # ── [US] 정규장 Wave 3 (04:30) ────────────────────────────────
        schedule.every().day.at(us["wave3"]).do(
            lambda: self._daemon_us_order(twap_wave=3)
        )

        # ── 로또 분석 (매주 수요일 10:00) ─────────────────────────────
        if _lotto_available:
            schedule.every().wednesday.at("10:00").do(_lotto_runner.run)

        logger.info("스케줄러 데몬 시작 (전 세션 + Layer 2 신호):")
        logger.info(f"  {sc['data_collect_time']}  데이터 수집")
        logger.info(f"  {sc.get('us_afterhours_time','05:10')}  [US] After-hours (10%)")
        logger.info(f"  {sc['us_signal_time']}  [US] 신호 생성 + 하락 매도")
        logger.info(f"  {sc['signal_gen_time']}  [KR] 신호 분리 + 하락 매도")
        logger.info(f"  {sc.get('kr_premarket_time','07:30')}  [KR] 장전 시간외 단일가 (15%)")
        logger.info(f"  {kr['wave1']}  [KR] Wave 1 (35%)")
        logger.info(f"  {sc.get('kr_intraday_update_1','10:00')}  [KR] Layer 2 업데이트 #1")
        logger.info(f"  {kr['wave2']}  [KR] Wave 2 (30%)")
        logger.info(f"  {sc.get('kr_intraday_update_2','13:00')}  [KR] Layer 2 업데이트 #2")
        logger.info(f"  {kr['wave3']}  [KR] Wave 3 (20%)")
        logger.info(f"  {sc.get('kr_afterclose_time','15:35')}  [KR] 장후 종가 (5%)")
        logger.info(f"  {sc['eod_record_time']}  EOD 기록")
        logger.info(f"  {sc.get('kr_aftersingle_time','16:30')}  [KR] 장후 단일가 (5%)")
        logger.info(f"  {sc.get('us_premarket_time','18:30')}  [US] Pre-market (15%)")
        logger.info(f"  {us['wave1']}  [US] Wave 1 (35%)")
        logger.info(f"  {sc.get('us_intraday_update_1','01:00')}  [US] Layer 2 업데이트 #1")
        logger.info(f"  {us['wave2']}  [US] Wave 2 (30%)")
        logger.info(f"  {sc.get('us_intraday_update_2','03:30')}  [US] Layer 2 업데이트 #2")
        logger.info(f"  {us['wave3']}  [US] Wave 3 (20%)")
        logger.info("  매주 수요일 10:00 [LOTTO] 로또 분석")

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
