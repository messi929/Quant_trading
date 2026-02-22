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
  - score ≥ 0.5 → aggressive (ask+0.3%, 10분 후 시장가)
  - score ≥ 0.2 → normal    (ask 기준, 10분 후 재조정)
  - score <  0.2 → patient  (중간값 지정가, 미체결 모니터링 없음)

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

# 왕복 거래비용 (수수료 0.2%×2 + 슬리피지 0.1%×2)
TRANSACTION_COST_RATE = 0.006

# TWAP 웨이브별 실행 비율 (합=1.0)
TWAP_FRACTIONS = {1: 0.40, 2: 0.35, 3: 0.25}

# 분할 실행 기준 금액 (이상이면 TWAP, 미만이면 Wave 1에서 전량)
TWAP_THRESHOLD = 500_000  # 50만원

# 미체결 재시도 대기 (초)
RETRY_WAIT_SECS = 300  # 5분


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

        _mode = self.cfg["broker"]["mode"]
        self.api_domestic = KISApi(mode=_mode, market_type="domestic")
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
        return self.api_domestic if market == "domestic" else self.api_overseas

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
                ba  = api.get_domestic_bid_ask(ticker)
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
                    p = float(api.get_domestic_price(ticker)["price"])
                else:
                    exch = exchange[:3] if exchange else "NAS"
                    p = float(api.get_overseas_price(ticker, exch)["price"])
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

    def _submit_limit_order(self, order: dict, price: float) -> dict:
        """지정가 주문."""
        api = self._get_api(order["market"])
        if order["market"] == "domestic":
            return api.order_domestic(
                ticker=order["ticker"],
                side=order["side"],
                qty=order["qty"],
                price=int(price),
                order_type="00",  # 지정가
            )
        else:
            return api.order_overseas(
                ticker=order["ticker"],
                side=order["side"],
                qty=order["qty"],
                price=round(price, 2),
                exchange=order.get("exchange", "NASD"),
            )

    def _submit_market_order(self, order: dict) -> dict:
        """시장가 주문 (최후 수단)."""
        logger.warning(
            f"시장가 전환: {order['ticker']} {order['side']} {order['qty']}주"
        )
        api = self._get_api(order["market"])
        if order["market"] == "domestic":
            return api.order_domestic(
                ticker=order["ticker"],
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

    def _execute_with_retry(self, order: dict, urgency: str) -> dict:
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
        result   = self._submit_limit_order(order, price)
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
            self.api.cancel_order(order_no, order["ticker"], order["side"], remaining_qty)
        except Exception as e:
            logger.warning(f"취소 실패 ({order_no}): {e}")
            return result

        # 가격 조정 (±0.5%)
        price2 = price * (1.005 if order["side"] == "buy" else 0.995)
        retry_order = {**order, "qty": remaining_qty}

        try:
            result2   = self._submit_limit_order(retry_order, price2)
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
                    order_no2, order["ticker"], order["side"], remaining2
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
    ) -> list[dict]:
        """모델 섹터 가중치로 리밸런싱 실행.

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

            buy_order = {**order, "qty": wave_qty}

            # 현금 부족 체크 (매도 실패한 경우만)
            if available_cash != float("inf"):
                try:
                    api = self._get_api(order["market"])
                    if order["market"] == "domestic":
                        p = float(api.get_domestic_price(order["ticker"])["price"])
                    else:
                        exch = order.get("exchange", "NAS")
                        p = float(api.get_overseas_price(
                            order["ticker"], exch[:3]
                        )["price"])
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
                result = self._execute_with_retry(buy_order, urgency)
                executed.append(result)
                self.logger.log_trade(result)
                if available_cash != float("inf"):
                    available_cash -= required
                logger.info(
                    f"매수: {order['ticker']} {wave_qty}주 "
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
        if self.execution_market in ("nasdaq", "split"):
            balance = self.api_overseas.get_overseas_balance()
            for pos in balance["positions"]:
                positions[pos["ticker"]] = {**pos, "sector": "unknown"}
        return positions

    def _get_portfolio_value(self, positions: dict) -> float:
        total = 0.0
        if self.execution_market in ("kospi", "split"):
            b = self.api_domestic.get_domestic_balance()
            total += b.get("total_eval", 0) + b.get("cash", 0)
        if self.execution_market in ("nasdaq", "split"):
            b = self.api_overseas.get_overseas_balance()
            total += b.get("total_eval", 0) + b.get("cash", 0)
        if total < self.total_capital * 0.1:
            logger.warning(f"포트폴리오 가치 낮음: {total:,.0f}")
            return self.total_capital
        return total

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
                        price_info = api.get_domestic_price(ticker)
                        exchange   = ""
                    else:
                        exchange   = tkr_info.get("exchange", "NASD")
                        price_info = api.get_overseas_price(
                            ticker, exchange[:3] if exchange else "NAS"
                        )
                    current_price = float(price_info["price"])
                except Exception as e:
                    logger.warning(f"{ticker} 현재가 조회 실패: {e}")
                    continue

                qty = int(per_ticker_amount / current_price)
                if qty < 1:
                    continue

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
        """현재 포지션 vs 목표 포지션 → 주문 목록."""
        orders      = []
        all_tickers = set(current.keys()) | set(targets.keys())

        for ticker in all_tickers:
            curr_qty   = current.get(ticker, {}).get("qty", 0)
            target_qty = targets.get(ticker, {}).get("target_qty", 0)

            if ticker not in targets:
                # 목표 없음 → 전량 매도
                if curr_qty > 0:
                    orders.append({
                        "ticker":  ticker,
                        "side":    "sell",
                        "qty":     curr_qty,
                        "sector":  current[ticker].get("sector", "unknown"),
                        "market":  "domestic" if ticker.isdigit() else "overseas",
                        "score":   0.0,
                        "target_amount": 0,
                    })
                continue

            diff_qty      = target_qty - curr_qty
            target_amount = targets[ticker]["target_amount"]
            curr_amount   = curr_qty * targets[ticker]["current_price"]
            amount_diff   = abs(target_amount - curr_amount) / max(target_amount, 1)

            # 허용 범위 내 → 스킵 (거래비용 절약)
            if amount_diff < self.rebal_threshold:
                continue

            score = targets[ticker].get("score", 0.3)
            base  = {
                "ticker":        ticker,
                "sector":        targets[ticker]["sector"],
                "market":        targets[ticker]["market"],
                "exchange":      targets[ticker].get("exchange", ""),
                "score":         score,
                "target_amount": target_amount,
            }

            if diff_qty > 0:
                orders.append({**base, "side": "buy",  "qty": diff_qty})
            elif diff_qty < 0:
                orders.append({**base, "side": "sell", "qty": abs(diff_qty)})

        return orders

    def execute_sell_check(self, sector_top_tickers: dict) -> list[dict]:
        """매일 06:30: 양수 신호 없는 보유 종목 즉시 매도."""
        positive_tickers = {
            t["ticker"]
            for tickers in sector_top_tickers.values()
            for t in tickers
        }
        current_positions = self._get_current_positions()

        sell_orders = []
        for ticker, pos in current_positions.items():
            if pos.get("qty", 0) <= 0:
                continue
            if ticker not in positive_tickers:
                sell_orders.append({
                    "ticker":  ticker,
                    "side":    "sell",
                    "qty":     pos["qty"],
                    "sector":  pos.get("sector", "unknown"),
                    "market":  "domestic" if ticker.isdigit() else "overseas",
                    "exchange": "",
                    "score":   0.0,
                    "target_amount": 0,
                })
                logger.info(f"하락 신호 매도 대상: {ticker} ({pos['qty']}주)")

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
                    "market":   "domestic" if ticker.isdigit() else "overseas",
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
