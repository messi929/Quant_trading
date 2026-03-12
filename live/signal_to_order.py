"""모델 신호 → 실제 주문 변환 (퀀트 트레이더 방식).

Phase A: 지정가 주문 + 미체결 시 재시도
  - 매수: ask 기준 / 매도: bid 기준 지정가
  - 5분 대기 후 미체결 → 가격 조정 재주문
  - aggressive: 추가 5분 후 미체결 → 시장가 전환

Phase B: TWAP 분할 실행 (50만원 이상 매수 주문)
  - Wave 1 (09:10): 40%
  - Wave 2 (11:00): 35%
  - Wave 3 (13:30): 25%
  - 매도는 항상 즉시 전량 (리스크 우선)

Phase C: 신호 강도 기반 긴급도
  - score ≥ 0.008 → aggressive (ask+0.3%, 10분 후 시장가)  [실제 score 상위 30%]
  - score ≥ 0.004 → normal    (ask 기준, 10분 후 재조정)   [실제 score 중위]
  - score <  0.004 → patient  (중간값 지정가, 미체결 모니터링 없음)

Phase D: 거래비용 임계값
  - 왕복 거래비용 ≈ 0.6% (수수료 0.2%×2 + 슬리피지 0.1%×2)
  - score < score_cost_threshold (기본 0.05) → 거래 스킵
"""

import time
import numpy as np
import yaml
from loguru import logger

from broker.kis_api import KISApi
from live.sector_instruments import get_sector_order
from tracking.trade_log import TradeLogger
from utils.ticker_utils import kis_code, is_domestic

# 왕복 거래비용 (수수료 0.2%×2 + 슬리피지 0.1%×2)
TRANSACTION_COST_RATE = 0.006

# TWAP 웨이브별 실행 비율 (합=1.0)
# None = 단일 실행 (100%, TWAP 없음)
TWAP_FRACTIONS = {None: 1.00, 1: 0.40, 2: 0.35, 3: 0.25}

# 분할 실행 기준 금액 (이상이면 TWAP, 미만이면 Wave 1에서 전량)
TWAP_THRESHOLD = 500_000  # 50만원

# 미체결 재시도 대기 (초)
RETRY_WAIT_SECS = 300  # 5분

# 세션별 리스크 파라미터
SESSION_RISK: dict = {
    "premarket_kr":  {"max_fraction": 0.15, "min_score": 0.008, "order_session": "premarket"},
    "afterclose_kr": {"max_fraction": 0.05, "min_score": 0.006, "order_session": "after_close"},
    "aftersingle_kr":{"max_fraction": 0.05, "min_score": 0.006, "order_session": "after_single"},
    "premarket_us":  {"max_fraction": 0.15, "min_score": 0.010, "order_session": "premarket"},
    "afterhours_us": {"max_fraction": 0.10, "min_score": 0.008, "order_session": "afterhours"},
}


def alpha_decay_scale(signal_age_hours: float, signal_type: str = "daily") -> float:
    """신호 생성 후 시간 경과에 따른 알파 붕괴 스케일.

    Args:
        signal_age_hours: 신호 생성 후 경과 시간
        signal_type: "daily"(24h 반감), "intraday"(4h 반감), "event"(1h 반감)

    Returns:
        0.1 ~ 1.0 스케일 팩터
    """
    half_lives = {"daily": 24.0, "intraday": 4.0, "event": 1.0}
    half_life = half_lives.get(signal_type, 24.0)
    scale = 0.5 ** (signal_age_hours / half_life)
    return max(float(scale), 0.1)


class OrderGenerator:
    """모델 신호를 퀀트 방식으로 실행."""

    def __init__(self, config_path: str = "config/live_config.yaml", twap_wave: int = 1):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        exec_cfg = self.cfg.get("execution", {})
        urg_cfg  = exec_cfg.get("urgency_thresholds", {})

        self.total_capital    = self.cfg["portfolio"]["total_capital"]
        self.rebal_threshold  = self.cfg["portfolio"]["rebalance_threshold"]
        self.max_sector_wt    = self.cfg["portfolio"]["max_sector_weight"]
        self.min_order_amount = self.cfg["portfolio"]["min_order_amount"]
        self.cash_buffer      = self.cfg["portfolio"]["cash_buffer"]
        self.execution_market = self.cfg["trading"]["execution_market"]

        # Phase B
        self.twap_wave      = twap_wave
        self.twap_threshold = exec_cfg.get("twap_threshold", TWAP_THRESHOLD)

        # Phase C
        self.urg_aggressive = urg_cfg.get("aggressive", 0.5)
        self.urg_normal     = urg_cfg.get("normal", 0.2)

        # Phase D
        self.score_cost_min = exec_cfg.get("score_cost_threshold", 0.05)

        # Kelly position sizing toggle (default: enabled)
        self.use_kelly_sizing = exec_cfg.get("use_kelly_sizing", True)
        self.kelly_lookback   = exec_cfg.get("kelly_lookback_days", 20)
        self._kelly_vol_cache: dict = {}  # {ticker: volatility}

        # 시장 충격 모델
        from strategy.market_impact import MarketImpactModel
        self._impact_model = MarketImpactModel()
        self.max_impact_pct = exec_cfg.get("max_impact_pct", 0.5)

        _mode = self.cfg["broker"]["mode"]
        self.paper_trading = self.cfg["broker"].get("paper_trading", False)
        if self.paper_trading:
            logger.info("paper_trading=True: 해외 주문은 가상 체결로 처리됩니다.")

        self.api_domestic = KISApi(mode=_mode, market_type="domestic")
        if self.paper_trading:
            # paper_trading=True: overseas KISApi init 생략 (토큰 403 방지)
            self.api_overseas = None
        else:
            try:
                self.api_overseas = KISApi(mode=_mode, market_type="overseas")
            except ValueError as e:
                logger.warning(f"해외 API 자격증명 미설정 → 국내 계좌로 대체: {e}")
                self.api_overseas = self.api_domestic

        # 하위 호환: self.api는 국내 기본 (pending/cancel 등 국내 전용 메서드용)
        self.api = self.api_domestic

        self.logger  = TradeLogger(self.cfg["logging"]["trade_log_db"])
        self.sectors = get_sector_order()

    def _get_api(self, market: str) -> KISApi:
        """market 문자열에 따라 적합한 API 인스턴스 반환."""
        if market == "domestic" or self.api_overseas is None:
            return self.api_domestic
        return self.api_overseas

    # ------------------------------------------------------------------
    # Phase C: 신호 강도 → 긴급도
    # ------------------------------------------------------------------

    def _get_urgency(self, score: float) -> str:
        if score >= self.urg_aggressive:
            return "aggressive"
        elif score >= self.urg_normal:
            return "normal"
        else:
            return "patient"

    # ------------------------------------------------------------------
    # Phase A: 호가 기반 지정가 계산
    # ------------------------------------------------------------------

    def _get_limit_price(
        self,
        ticker: str,
        market: str,
        side: str,
        urgency: str,
        exchange: str = "",
    ) -> float:
        """긴급도에 따른 지정가 반환.

        aggressive 매수: ask+0.3%  (즉시 체결 우선)
        normal     매수: ask        (호가 기준)
        patient    매수: (bid+ask)/2 (좋은 가격 대기)
        매도는 반대 방향.
        """
        api = self._get_api(market)
        try:
            if market == "domestic":
                ba  = api.get_domestic_bid_ask(kis_code(ticker))
                bid = float(ba["bid"])
                ask = float(ba["ask"])
            else:
                exch = exchange[:3] if exchange else "NAS"
                p    = api.get_overseas_price(ticker, exch)["price"]
                spread = p * 0.0005
                bid, ask = p - spread, p + spread
        except Exception as e:
            logger.warning(f"{ticker} 호가 조회 실패 — 현재가 대체: {e}")
            try:
                if market == "domestic":
                    p = float(api.get_domestic_price(kis_code(ticker))["price"])
                else:
                    import yfinance as yf
                    p = yf.Ticker(ticker).fast_info.last_price
                    logger.info(f"{ticker} 호가 yfinance 폴백: ${p:.2f}")
            except Exception:
                raise
            bid = ask = p

        if side == "buy":
            if urgency == "aggressive":
                return ask * 1.003        # ask+0.3%: 즉시 체결 우선
            elif urgency == "normal":
                return ask                # ask: 바로 체결
            else:
                return (bid + ask) / 2   # 중간값: 좋은 가격 대기
        else:
            if urgency == "aggressive":
                return bid * 0.997        # bid-0.3%
            elif urgency == "normal":
                return bid                # bid: 바로 체결
            else:
                return (bid + ask) / 2   # 중간값

    # ------------------------------------------------------------------
    # Phase A: 지정가 / 시장가 주문 제출
    # ------------------------------------------------------------------

    def _submit_limit_order(self, order: dict, price: float, session: str = "regular") -> dict:
        """지정가 주문."""
        api = self._get_api(order["market"])
        if order["market"] == "domestic":
            return api.order_domestic(
                ticker=kis_code(order["ticker"]),  # KIS: 6자리 코드만 허용
                side=order["side"],
                qty=order["qty"],
                price=int(price),
                order_type="00",  # 지정가
                session=session,
            )
        else:
            return api.order_overseas(
                ticker=order["ticker"],
                side=order["side"],
                qty=order["qty"],
                price=round(price, 2),
                exchange=order.get("exchange", "NASD"),
                session=session,
            )

    def _submit_market_order(self, order: dict) -> dict:
        """시장가 주문 (최후 수단)."""
        logger.warning(
            f"시장가 전환: {order['ticker']} {order['side']} {order['qty']}주"
        )
        api = self._get_api(order["market"])
        if order["market"] == "domestic":
            return api.order_domestic(
                ticker=kis_code(order["ticker"]),  # KIS: 6자리 코드만 허용
                side=order["side"],
                qty=order["qty"],
                price=0,
                order_type="01",  # 시장가
            )
        else:
            # 해외: KIS API에서 현재가±2% 지정가로 처리
            return api.order_overseas(
                ticker=order["ticker"],
                side=order["side"],
                qty=order["qty"],
                price=0,
                exchange=order.get("exchange", "NASD"),
            )

    # ------------------------------------------------------------------
    # Phase A: 지정가 → 미체결 시 재시도
    # ------------------------------------------------------------------

    def _execute_with_retry(self, order: dict, urgency: str, session: str = "regular") -> dict:
        """
        1차: 지정가 주문
        5분 후: 미체결 확인
          → 미체결 → 취소 후 가격 조정 재주문
             aggressive: 추가 5분 후 미체결 → 시장가 전환
             normal:     가격 조정 후 종료 (다음날 재시도)
             patient:    모니터링 없음 (당일 미체결 → 자동 취소)
        """
        price = self._get_limit_price(
            order["ticker"], order["market"], order["side"], urgency,
            order.get("exchange", ""),
        )
        result   = self._submit_limit_order(order, price, session=session)
        order_no = result.get("order_no", "")

        logger.info(
            f"[{urgency}] {order['side'].upper()} {order['ticker']} "
            f"{order['qty']}주 @ {price:.0f} (주문번호={order_no})"
        )

        # patient: 모니터링 없이 반환 (당일 미체결 → 장 마감 후 자동 취소)
        if urgency == "patient" or not order_no:
            return result

        # ── 5분 대기 후 미체결 확인 ──────────────────────────────────
        time.sleep(RETRY_WAIT_SECS)

        try:
            pending = self.api.get_pending_orders()
        except Exception as e:
            logger.warning(f"미체결 조회 실패: {e}")
            return result

        pend = next((p for p in pending if p["order_no"] == order_no), None)
        if pend is None or pend["remaining_qty"] == 0:
            logger.info(f"[{urgency}] {order['ticker']}: 5분 내 전량 체결")
            return result

        remaining_qty = pend["remaining_qty"]
        logger.info(
            f"[{urgency}] {order['ticker']}: {remaining_qty}주 미체결 → 취소 후 가격 조정"
        )

        # 기존 주문 취소
        try:
            self.api.cancel_order(order_no, kis_code(order["ticker"]), order["side"], remaining_qty)
        except Exception as e:
            logger.warning(f"취소 실패 ({order_no}): {e}")
            return result

        # 가격 조정 (±0.5%)
        price2 = price * (1.005 if order["side"] == "buy" else 0.995)
        retry_order = {**order, "qty": remaining_qty}

        try:
            result2   = self._submit_limit_order(retry_order, price2, session=session)
            order_no2 = result2.get("order_no", "")
            logger.info(
                f"[{urgency}] {order['ticker']}: 재주문 @ {price2:.0f} "
                f"(주문번호={order_no2})"
            )
        except Exception as e:
            logger.error(f"재주문 실패 [{order['ticker']}]: {e}")
            return result

        # normal: 한 번 재조정 후 종료
        if urgency == "normal":
            return result2

        # ── aggressive: 추가 5분 대기 후 미체결 → 시장가 ─────────────
        time.sleep(RETRY_WAIT_SECS)

        if not order_no2:
            return result2

        try:
            pending2 = self.api.get_pending_orders()
            pend2    = next((p for p in pending2 if p["order_no"] == order_no2), None)
        except Exception:
            return result2

        if pend2 and pend2["remaining_qty"] > 0:
            remaining2 = pend2["remaining_qty"]
            try:
                self.api.cancel_order(
                    order_no2, kis_code(order["ticker"]), order["side"], remaining2
                )
                return self._submit_market_order({**retry_order, "qty": remaining2})
            except Exception as e:
                logger.error(f"시장가 전환 실패 [{order['ticker']}]: {e}")

        return result2

    # ------------------------------------------------------------------
    # 메인 진입점
    # ------------------------------------------------------------------

    def execute_rebalance(
        self,
        sector_weights: np.ndarray,
        sector_top_tickers: dict = None,
        market_filter: str = None,
        layer2_scales: dict = None,
    ) -> list[dict]:
        """모델 섹터 가중치로 리밸런싱 실행.

        Args:
            market_filter: None=전체, "domestic"=국내만, "overseas"=해외만
            layer2_scales: Layer 2 장중 신호 스케일 {ticker_normalized: scale}

        Phase B: twap_wave에 따라 매수 수량 분할
          - 매도는 항상 즉시 전량 (리스크 우선)
          - 50만원 미만 소액 매수는 Wave 1에서만 전량 실행
        """
        weights           = self._validate_weights(sector_weights)
        current_positions = self._get_current_positions()
        portfolio_value   = self._get_portfolio_value(current_positions)
        target_positions  = self._compute_target_positions(
            weights, portfolio_value, sector_top_tickers
        )
        # market_filter 적용: 지정 시장 종목만 주문
        if market_filter:
            target_positions = {
                t: v for t, v in target_positions.items()
                if v.get("market") == market_filter
            }
        orders = self._compute_orders(current_positions, target_positions)

        if not orders:
            logger.info(f"[Wave{self.twap_wave}] 리밸런싱 불필요")
            return []

        if self._is_daily_loss_exceeded(current_positions):
            logger.warning("일간 손실 한도 초과 → 리밸런싱 중단")
            return []

        sell_orders = [o for o in orders if o["side"] == "sell"]
        buy_orders  = [o for o in orders if o["side"] == "buy"]
        executed    = []
        failed_sell_tickers = set()

        # ── 매도 실행 (TWAP 없음 — 리스크 우선, 항상 전량) ─────────────
        for order in sell_orders:
            urgency = self._get_urgency(order.get("score", 0.3))
            try:
                result = self._execute_with_retry(order, urgency)
                executed.append(result)
                self.logger.log_trade(result)
            except Exception as e:
                failed_sell_tickers.add(order["ticker"])
                logger.error(f"매도 실패 [{order['ticker']}]: {e}")

        if failed_sell_tickers:
            logger.warning(f"매도 실패: {failed_sell_tickers} → 익일 재시도")

        # ── 매수 후 가용 현금 확인 ──────────────────────────────────────
        if failed_sell_tickers:
            try:
                if self.execution_market in ("kospi", "split"):
                    raw = self.api_domestic.get_domestic_balance()
                else:
                    raw = self.api_overseas.get_overseas_balance()
                available_cash = raw.get("cash", 0)
                logger.info(f"매도 실패 후 가용 현금: {available_cash:,.0f}")
            except Exception:
                available_cash = float("inf")
        else:
            available_cash = float("inf")

        # ── Phase B: 매수 TWAP 분할 ─────────────────────────────────────
        wave_fraction = TWAP_FRACTIONS.get(self.twap_wave, 1.0)

        for order in buy_orders:
            target_amount = order.get("target_amount", 0)
            urgency       = self._get_urgency(order.get("score", 0.3))

            # 소액 주문: Wave 1에서만 전량 실행, 이후 웨이브는 스킵
            if target_amount < self.twap_threshold:
                if self.twap_wave != 1:
                    continue
                wave_qty = order["qty"]
            else:
                # 대형 주문: 웨이브 비율만큼만 실행
                wave_qty = max(1, int(order["qty"] * wave_fraction))

            # Layer 2 스케일 적용 (장중 모멘텀 반영)
            if layer2_scales:
                from utils.ticker_utils import kis_code as _kc
                _norm = _kc(order["ticker"])
                _scale = layer2_scales.get(_norm, layer2_scales.get(order["ticker"], 1.0))
                wave_qty = max(1, int(wave_qty * min(_scale, 1.5)))

            buy_order = {**order, "qty": wave_qty}

            # 현금 부족 체크 (매도 실패한 경우만)
            if available_cash != float("inf"):
                try:
                    api = self._get_api(order["market"])
                    if order["market"] == "domestic":
                        p = float(api.get_domestic_price(order["ticker"])["price"])
                    else:
                        try:
                            exch = order.get("exchange", "NAS")
                            p = float(api.get_overseas_price(
                                order["ticker"], exch[:3]
                            )["price"])
                        except Exception:
                            import yfinance as yf
                            p = yf.Ticker(order["ticker"]).fast_info.last_price
                    required = p * wave_qty
                except Exception:
                    required = 0

                if required > available_cash:
                    logger.warning(
                        f"매수 스킵 [{order['ticker']}]: "
                        f"필요 {required:,.0f} > 가용 {available_cash:,.0f}"
                    )
                    continue

            try:
                # overseas: sandbox USD 미지원 → yfinance 가상 체결
                # domestic: KIS sandbox에 실제 주문 (모의투자 정상 작동)
                if self.paper_trading and buy_order["market"] == "overseas":
                    result = self._execute_paper_trade(buy_order)
                else:
                    result = self._execute_with_retry(buy_order, urgency)
                executed.append(result)
                self.logger.log_trade(result)
                if available_cash != float("inf"):
                    available_cash -= required
                logger.info(
                    f"{'[PAPER] ' if result.get('mode') == 'paper' else ''}매수: "
                    f"{order['ticker']} {wave_qty}주 "
                    f"[Wave{self.twap_wave} {wave_fraction:.0%} / {urgency}]"
                )
            except Exception as e:
                logger.error(f"매수 실패 [{order['ticker']}]: {e}")

        logger.info(
            f"[Wave{self.twap_wave}] 리밸런싱 완료: "
            f"{len(executed)}/{len(orders)} 실행 "
            f"(매도 실패 {len(failed_sell_tickers)}건)"
        )
        return executed

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        weights   = np.clip(weights, 0, self.max_sector_wt)
        investable = 1.0 - self.cash_buffer
        total      = weights.sum()
        if total > investable:
            weights = weights / total * investable
        elif total < 1e-8:
            weights = np.zeros(len(weights))
        return weights

    def _get_current_positions(self) -> dict[str, dict]:
        positions = {}
        if self.execution_market in ("kospi", "split"):
            balance = self.api_domestic.get_domestic_balance()
            for pos in balance["positions"]:
                positions[pos["ticker"]] = {**pos, "sector": "unknown"}
        if self.execution_market in ("nasdaq", "split") and not self.paper_trading:
            balance = self.api_overseas.get_overseas_balance()
            for pos in balance["positions"]:
                positions[pos["ticker"]] = {**pos, "sector": "unknown"}
        return positions

    def _get_portfolio_value(self, positions: dict) -> float:
        # KIS API total_eval(tot_evlu_amt) = 주식평가 + 예수금 포함 전체 계좌가치
        # → cash를 따로 더하면 이중 계산됨
        total = 0.0
        if self.execution_market in ("kospi", "split"):
            b = self.api_domestic.get_domestic_balance()
            total += b.get("total_eval", 0)   # 현금+주식 합산값
        if self.execution_market in ("nasdaq", "split") and not self.paper_trading:
            try:
                b = self.api_overseas.get_overseas_balance()
                total += b.get("total_eval", 0)
            except Exception as e:
                logger.warning(f"해외 잔고 조회 실패 (sandbox 미지원일 수 있음): {e}")
        if total < self.total_capital * 0.1:
            logger.warning(f"포트폴리오 가치 낮음: {total:,.0f}")
            return self.total_capital
        return total

    def _get_fx_rate(self) -> float:
        """USD/KRW 환율 조회 (캐시, 세션당 1회 fetch)."""
        if hasattr(self, "_fx_rate_cache"):
            return self._fx_rate_cache
        try:
            import yfinance as yf
            rate = float(yf.Ticker("USDKRW=X").fast_info.last_price)
            if rate > 500:  # 정상 범위 확인 (500~2000)
                self._fx_rate_cache = rate
                logger.info(f"USD/KRW 환율: {rate:.1f}")
                return rate
        except Exception as e:
            logger.warning(f"환율 조회 실패 → 1,350 사용: {e}")
        self._fx_rate_cache = 1350.0
        return self._fx_rate_cache

    def _execute_paper_trade(self, order: dict) -> dict:
        """Paper trading: yfinance 현재가로 가상 체결 (KIS API 미호출)."""
        import yfinance as yf
        from datetime import datetime as dt
        ticker = order["ticker"]
        try:
            price = float(yf.Ticker(ticker).fast_info.last_price)
        except Exception:
            price = order.get("target_amount", 0) / max(order["qty"], 1)
        return {
            "timestamp": dt.now().isoformat(),
            "ticker":    ticker,
            "side":      order["side"],
            "qty":       order["qty"],
            "price":     price,
            "amount":    order["qty"] * price,
            "sector":    order.get("sector", ""),
            "order_no":  f"PAPER_{ticker}_{int(time.time())}",
            "mode":      "paper",
            "note":      f"wave{self.twap_wave}",
        }

    def _kelly_scale(self, ticker: str, score: float, market: str) -> float:
        """Half-Kelly 스케일 팩터 계산.

        Args:
            ticker: 종목 코드
            score: 모델 신호 강도 (0 ~ 0.05 범위)
            market: "domestic" or "overseas"

        Returns:
            0.3 ~ 1.5 범위 스케일 팩터
        """
        if not self.use_kelly_sizing:
            return 1.0

        # 변동성 조회 (캐시 사용)
        norm = kis_code(ticker)
        vol = self._kelly_vol_cache.get(norm)

        if vol is None:
            try:
                import yfinance as yf
                hist = yf.Ticker(ticker).history(period=f"{self.kelly_lookback}d")
                if len(hist) >= 5:
                    vol = hist["Close"].pct_change().dropna().std()
                    self._kelly_vol_cache[norm] = vol
            except Exception:
                pass

        if vol is None or vol < 1e-8:
            return 1.0

        # Half-Kelly: f = signal_edge / variance (scaled down by 0.5)
        # signal_edge: score 범위 0.001~0.011 → 정규화
        normalized_score = min(score / 0.008, 2.0)  # 0.008 = 기준 score (aggressive 임계값)
        kelly_f = 0.5 * normalized_score / (vol * 100 + 1e-8)

        # 0.3 ~ 1.5로 클리핑 (과도한 배율 방지)
        return float(np.clip(kelly_f, 0.3, 1.5))

    def _compute_target_positions(
        self,
        weights: np.ndarray,
        portfolio_value: float,
        sector_top_tickers: dict = None,
    ) -> dict[str, dict]:
        """목표 포지션 계산.

        execution_market에 따라 투자 종목 결정:
          - kospi  : KOSPI_ETF_MAP 사용 (KODEX 시리즈) — 국내 시세 조회
          - nasdaq : NASDAQ_ETF_MAP 사용 (XL 시리즈)   — 해외 시세 조회
          - split  : 모델 top-ticker 그대로 사용 (Phase D cost 필터 적용)
        """
        from live.sector_instruments import KOSPI_ETF_MAP, NASDAQ_ETF_MAP

        sector_top_tickers = sector_top_tickers or {}
        targets = {}

        for i, sector in enumerate(self.sectors):
            if weights[i] < 1e-4:
                continue

            sector_amount = portfolio_value * weights[i]
            if sector_amount < self.min_order_amount:
                continue

            # ── execution_market에 따라 투자 종목 결정 ──────────────────
            if self.execution_market == "kospi":
                etf_info = KOSPI_ETF_MAP.get(sector)
                if not etf_info:
                    continue
                ticker_list = [{
                    "ticker":   etf_info["ticker"],
                    "market":   "domestic",
                    "exchange": "",
                    "score":    float(weights[i]),
                }]

            elif self.execution_market == "nasdaq":
                etf_info = NASDAQ_ETF_MAP.get(sector)
                if not etf_info:
                    continue
                ticker_list = [{
                    "ticker":   etf_info["ticker"],
                    "market":   "overseas",
                    "exchange": etf_info.get("exchange", "NYSE"),
                    "score":    float(weights[i]),
                }]

            else:  # split: 모델 top-ticker 그대로 사용
                top_tickers = sector_top_tickers.get(sector, [])
                if not top_tickers:
                    continue
                ticker_list = top_tickers

            per_ticker_amount = sector_amount / len(ticker_list)
            if per_ticker_amount < self.min_order_amount:
                ticker_list       = ticker_list[:1]
                per_ticker_amount = sector_amount

            for tkr_info in ticker_list:
                ticker = tkr_info["ticker"]
                market = tkr_info["market"]
                score  = float(tkr_info.get("score", 0.0))

                # Phase D: split 모드에서만 score 임계값 필터 적용 (ETF는 항상 거래)
                if self.execution_market == "split" and score < self.score_cost_min:
                    logger.debug(
                        f"스킵 [{ticker}]: score={score:.3f} < "
                        f"임계값={self.score_cost_min:.3f} (거래비용 미충족)"
                    )
                    continue

                try:
                    api = self._get_api(market)
                    if market == "domestic":
                        # KIS domestic API는 6자리 코드만 허용 (.KS/.KQ suffix 제거)
                        kis_ticker = kis_code(ticker)
                        try:
                            price_info    = api.get_domestic_price(kis_ticker)
                            exchange      = ""
                            current_price = float(price_info["price"])
                        except Exception:
                            import yfinance as yf
                            _raw = yf.Ticker(ticker if "." in ticker else f"{ticker}.KS").fast_info.last_price
                            if not _raw or _raw <= 0:
                                raise ValueError(f"yfinance 가격 없음: {ticker}")
                            current_price = float(_raw)
                            exchange      = ""
                            logger.info(f"{ticker} KIS 시세 불가 → yfinance 폴백: ₩{current_price:,.0f}")
                    else:
                        exchange   = tkr_info.get("exchange", "NASD")
                        try:
                            price_info    = api.get_overseas_price(
                                ticker, exchange[:3] if exchange else "NAS"
                            )
                            current_price = float(price_info["price"])
                        except Exception:
                            import yfinance as yf
                            _raw = yf.Ticker(ticker).fast_info.last_price
                            if not _raw or _raw <= 0:
                                raise ValueError(f"yfinance 가격 없음: {ticker}")
                            current_price = float(_raw)
                            logger.info(f"{ticker} KIS 시세 불가 → yfinance 폴백: ${current_price:.2f}")
                except Exception as e:
                    logger.warning(f"{ticker} 현재가 조회 실패: {e}")
                    continue

                if current_price <= 0:
                    logger.warning(f"{ticker} 현재가 0 또는 음수 → 스킵")
                    continue

                # paper_trading + overseas: KRW 금액 → USD 환산 후 수량 계산
                if self.paper_trading and market == "overseas":
                    amount_for_qty = per_ticker_amount / self._get_fx_rate()
                else:
                    amount_for_qty = per_ticker_amount
                qty = int(amount_for_qty / current_price)
                if qty < 1:
                    continue

                # Half-Kelly 포지션 조정 (신호 강도 반영)
                if self.use_kelly_sizing and score > 0:
                    kelly_s = self._kelly_scale(ticker, score, market)
                    qty = max(1, int(qty * kelly_s))
                    if kelly_s != 1.0:
                        logger.debug(
                            f"Kelly sizing {ticker}: score={score:.4f}, "
                            f"kelly_scale={kelly_s:.2f}, qty={qty}"
                        )

                # 시장 충격 조정 (거래량 대비 주문 크기 제한)
                try:
                    import yfinance as yf
                    _vol_data = yf.Ticker(ticker if '.' in ticker else f"{ticker}.KS" if market == 'domestic' else ticker)
                    _hist = _vol_data.history(period="20d")
                    if len(_hist) >= 5:
                        avg_vol = _hist["Volume"].mean()
                        daily_vol_pct = _hist["Close"].pct_change().dropna().std()
                        tier = self._impact_model.get_volume_tier(avg_vol, "KOSPI" if market == "domestic" else "NASDAQ")
                        qty = self._impact_model.adjust_order_size(
                            qty, avg_vol, daily_vol_pct, current_price,
                            max_impact_pct=self.max_impact_pct,
                            market_cap_tier=tier,
                            market="KOSPI" if market == "domestic" else "NASDAQ",
                        )
                except Exception:
                    pass  # 충격 모델 실패 시 원래 수량 유지

                if ticker in targets:
                    targets[ticker]["target_qty"]    += qty
                    targets[ticker]["target_amount"] += per_ticker_amount
                    targets[ticker]["score"]          = max(targets[ticker]["score"], score)
                else:
                    targets[ticker] = {
                        "sector":        sector,
                        "target_amount": per_ticker_amount,
                        "target_qty":    qty,
                        "current_price": current_price,
                        "market":        market,
                        "exchange":      exchange,
                        "score":         score,
                    }

        return targets

    def _compute_orders(
        self,
        current: dict[str, dict],
        targets: dict[str, dict],
    ) -> list[dict]:
        """현재 포지션 vs 목표 포지션 → 주문 목록.

        KIS balance API는 6자리 숫자 ticker 반환 (예: "005930").
        sector_top_tickers(parquet 기준)는 ".KS"/".KQ" suffix 포함 (예: "005930.KS").
        suffix 제거 후 매칭하여 동일 종목 중복 주문(매도+재매수) 방지.
        """
        orders = []

        # 정규화 키(suffix 제거) → 원본 키 매핑
        # current: KIS balance → 이미 정규화된 6자리 코드 또는 US 심볼
        # targets: parquet → ".KS"/".KQ" suffix 포함 가능
        norm_to_curr = {k: k for k in current}
        norm_to_tgt  = {kis_code(k): k for k in targets}

        all_norm = set(norm_to_curr.keys()) | set(norm_to_tgt.keys())

        for norm in all_norm:
            curr_key = norm_to_curr.get(norm)
            tgt_key  = norm_to_tgt.get(norm)

            curr_qty   = current[curr_key].get("qty", 0) if curr_key else 0
            tgt_info   = targets[tgt_key] if tgt_key else {}
            target_qty = tgt_info.get("target_qty", 0)

            if tgt_key is None:
                # 목표 없음 → 전량 매도 (curr_key 사용 — KIS 6자리 형식 유지)
                if curr_qty > 0:
                    ticker = curr_key
                    orders.append({
                        "ticker":        ticker,
                        "side":          "sell",
                        "qty":           curr_qty,
                        "sector":        current[curr_key].get("sector", "unknown"),
                        "market":        "domestic" if is_domestic(ticker) else "overseas",
                        "score":         0.0,
                        "target_amount": 0,
                    })
                continue

            # tgt_key(suffix 포함) 기준으로 주문 생성 — market/exchange 정보 보유
            ticker        = tgt_key
            diff_qty      = target_qty - curr_qty
            target_amount = tgt_info["target_amount"]
            curr_amount   = curr_qty * tgt_info.get("current_price", 0)
            amount_diff   = abs(target_amount - curr_amount) / max(target_amount, 1)

            # 허용 범위 내 → 스킵 (거래비용 절약)
            if amount_diff < self.rebal_threshold:
                continue

            score = tgt_info.get("score", 0.3)
            base  = {
                "ticker":        ticker,
                "sector":        tgt_info["sector"],
                "market":        tgt_info["market"],
                "exchange":      tgt_info.get("exchange", ""),
                "score":         score,
                "target_amount": target_amount,
            }

            if diff_qty > 0:
                orders.append({**base, "side": "buy", "qty": diff_qty})
            elif diff_qty < 0:
                # 매도 시: curr_key(6자리) 사용 — KIS order API 호환
                sell_ticker = curr_key if curr_key else ticker
                orders.append({
                    **base,
                    "ticker": sell_ticker,
                    "side":   "sell",
                    "qty":    abs(diff_qty),
                })

        return orders

    def execute_sell_check(
        self,
        sector_top_tickers: dict,
        market_filter: str = None,
    ) -> list[dict]:
        """보유 종목 중 양수 신호 없는 종목 즉시 매도.

        Args:
            sector_top_tickers: 섹터별 top 종목 (신호 있는 종목 목록)
            market_filter: None=전체, "domestic"=국내만, "overseas"=해외만

        KIS balance tickers: 6자리 숫자 (예: "005930")
        sector_top_tickers: parquet 형식 (예: "005930.KS")
        → suffix 제거 후 비교하여 매도 오판 방지
        """
        market_label = f"[{market_filter}] " if market_filter else ""
        # ".KS"/".KQ" suffix 제거하여 KIS 잔고 티커 형식과 통일
        positive_tickers = {
            kis_code(t["ticker"])
            for tickers in sector_top_tickers.values()
            for t in tickers
        }
        current_positions = self._get_current_positions()

        sell_orders = []
        for ticker, pos in current_positions.items():
            if pos.get("qty", 0) <= 0:
                continue
            # 시장 필터: 6자리 숫자 → domestic, 그 외 → overseas
            is_domestic_flag = is_domestic(ticker)
            if market_filter == "domestic" and not is_domestic_flag:
                continue  # 해외 포지션은 이번 체크에서 스킵
            if market_filter == "overseas" and is_domestic_flag:
                continue  # 국내 포지션은 이번 체크에서 스킵
            # current_positions 키는 KIS 6자리 형식 → 정규화 없이 직접 비교
            if kis_code(ticker) not in positive_tickers:
                sell_orders.append({
                    "ticker":        ticker,
                    "side":          "sell",
                    "qty":           pos["qty"],
                    "sector":        pos.get("sector", "unknown"),
                    "market":        "domestic" if is_domestic_flag else "overseas",
                    "exchange":      "",
                    "score":         0.0,
                    "target_amount": 0,
                })
                logger.info(f"{market_label}하락 신호 매도 대상: {ticker} ({pos['qty']}주)")

        if not sell_orders:
            logger.info("하락 신호 종목 없음")
            return []

        if self._is_daily_loss_exceeded(current_positions):
            logger.warning("일간 손실 한도 초과 → 매도 중단")
            return []

        executed = []
        failed   = []
        for order in sell_orders:
            # 하락 신호 매도: 빨리 처리 (aggressive)
            try:
                result = self._execute_with_retry(order, "aggressive")
                executed.append(result)
                self.logger.log_trade(result)
            except Exception as e:
                failed.append(order["ticker"])
                logger.error(f"매도 실패 [{order['ticker']}]: {e}")

        if failed:
            logger.warning(f"매도 실패 {len(failed)}건: {failed} → 09:10 재시도")

        logger.info(f"하락 신호 매도: {len(executed)}/{len(sell_orders)}건")
        return executed

    def execute_extended_hours(
        self,
        sector_weights: np.ndarray,
        sector_top_tickers: dict,
        session: str,
        signal_age_hours: float = 0.0,
        layer2_scales: dict = None,
        market_filter: str = None,
    ) -> list[dict]:
        """시간외 세션 실행 (장전/장후/프리마켓/에프터마켓).

        Args:
            session: "premarket_kr", "afterclose_kr", "aftersingle_kr",
                     "premarket_us", "afterhours_us"
            signal_age_hours: 신호 생성 후 경과 시간 (alpha decay 계산용)
            layer2_scales: Layer 2 스케일 팩터 {ticker_normalized: scale}
            market_filter: "domestic" or "overseas"

        특징:
            - 매도 없음 (추가 매수만)
            - 강한 신호(min_score 이상) 종목만
            - Alpha decay로 오래된 신호 자동 축소
        """
        risk = SESSION_RISK.get(session)
        if risk is None:
            logger.warning(f"알 수 없는 session: {session}")
            return []

        max_fraction  = risk["max_fraction"]
        min_score     = risk["min_score"]
        order_session = risk["order_session"]

        # Alpha decay 적용
        decay              = alpha_decay_scale(signal_age_hours)
        effective_fraction = max_fraction * decay

        market_label = f"[{session}|decay={decay:.2f}|frac={effective_fraction:.1%}]"
        logger.info(f"=== {market_label} 시간외 실행 시작 ===")

        # min_score 이상 강한 신호만 필터
        filtered_tickers: dict = {}
        for sector, tickers in sector_top_tickers.items():
            kept = [t for t in tickers if float(t.get("score", 0)) >= min_score]
            if kept:
                filtered_tickers[sector] = kept

        if not filtered_tickers:
            logger.info(f"{market_label} min_score={min_score:.3f} 통과 종목 없음 → 스킵")
            return []

        # 목표 포지션: effective_fraction 비율만큼
        weights          = self._validate_weights(sector_weights)
        scaled_weights   = weights * effective_fraction
        current_positions = self._get_current_positions()
        portfolio_value  = self._get_portfolio_value(current_positions)

        target_positions = self._compute_target_positions(
            scaled_weights, portfolio_value, filtered_tickers
        )

        # Layer 2 스케일 적용
        if layer2_scales:
            for ticker in list(target_positions.keys()):
                from utils.ticker_utils import kis_code as _kc
                _norm  = _kc(ticker)
                _scale = layer2_scales.get(_norm, layer2_scales.get(ticker, 1.0))
                if _scale <= 0:
                    del target_positions[ticker]
                else:
                    tgt = target_positions[ticker]
                    tgt["target_qty"] = max(1, int(tgt["target_qty"] * min(_scale, 1.5)))

        # 시장 필터
        if market_filter:
            target_positions = {
                t: v for t, v in target_positions.items()
                if v.get("market") == market_filter
            }

        # 매수 주문만 (시간외는 매도 없음)
        orders     = self._compute_orders(current_positions, target_positions)
        buy_orders = [o for o in orders if o["side"] == "buy"]

        if not buy_orders:
            logger.info(f"{market_label} 실행할 매수 주문 없음")
            return []

        if self._is_daily_loss_exceeded(current_positions):
            logger.warning(f"{market_label} 일간 손실 한도 초과 → 시간외 거래 중단")
            return []

        executed = []
        for order in buy_orders:
            urgency = self._get_urgency(order.get("score", 0))
            try:
                if self.paper_trading and order["market"] == "overseas":
                    result = self._execute_paper_trade(order)
                else:
                    result = self._execute_with_retry(order, urgency, session=order_session)
                executed.append(result)
                self.logger.log_trade(result)
                logger.info(
                    f"{market_label} {'[PAPER] ' if result.get('mode') == 'paper' else ''}"
                    f"매수: {order['ticker']} {order['qty']}주 [{order_session}]"
                )
            except Exception as e:
                logger.error(f"{market_label} 매수 실패 [{order['ticker']}]: {e}")

        logger.info(f"=== {market_label} 완료: {len(executed)}/{len(buy_orders)}건 ===")
        return executed

    def _is_daily_loss_exceeded(self, positions: dict) -> bool:
        daily_limit = self.cfg["risk"]["daily_loss_limit"]
        pnl         = self.logger.get_today_pnl_pct()
        if pnl < -daily_limit:
            logger.warning(f"일간 손실 {pnl:.2%} (한도 {daily_limit:.2%})")
            return True
        return False

    def emergency_liquidate(self):
        """전량 청산 (긴급)."""
        logger.warning("긴급 청산 시작!")
        positions = self._get_current_positions()
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                order = {
                    "ticker":   ticker,
                    "side":     "sell",
                    "qty":      pos["qty"],
                    "market":   "domestic" if is_domestic(ticker) else "overseas",
                    "sector":   pos.get("sector", "unknown"),
                    "score":    0.0,
                    "exchange": "",
                    "target_amount": 0,
                }
                try:
                    result = self._execute_with_retry(order, "aggressive")
                    self.logger.log_trade(result)
                except Exception as e:
                    logger.error(f"청산 실패 [{ticker}]: {e}")
        logger.warning("긴급 청산 완료")
