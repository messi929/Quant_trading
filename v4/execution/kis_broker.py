"""V4 KIS 국내 broker 래퍼 — broker/kis_api.py(KISApi domestic) 재사용.

KIS sandbox 모의투자 계좌(국내, ₩1억)에 실제 주문/잔고 연동. 계좌 상태는 KIS가
추적(V3 yfinance 시뮬과 달리) → 실자본 전환 시 mode="prod"만 변경.

dry_run=True: 주문 API 미호출, 의도만 로그 (live 가격으로 파이프라인 검증용).

주의: 실주문 체결은 KR 장중에만. 토큰/가격 조회는 장외에도 가능.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from broker.kis_api import KISApi

# KIS sandbox 가 빈번히 던지는 일시적(transient) 에러 — 재시도하면 대개 성공.
# 비일시적(계좌차단 40910000, 매매불가 종목 등)은 여기 없으므로 즉시 실패한다.
_TRANSIENT_MARKERS = (
    "500 Server Error",
    "502 ", "503 ", "504 ",
    "Connection aborted",
    "RemoteDisconnected",
    "Read timed out",
    "Max retries exceeded",
    "Temporarily",
)


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(m in msg for m in _TRANSIENT_MARKERS)


class KisKoreaBroker:
    """KISApi 국내 얇은 래퍼. price_krw / buy(amount) / sell(qty) / balance."""

    MAX_ORDER_RETRIES = 3      # 일시적 에러 시 총 시도 횟수
    RETRY_WAIT_S = 1.0         # 재시도 간 대기(초), attempt 비례 backoff

    def __init__(self, mode: str = "sandbox", dry_run: bool = False, log_dir: str = "v4/logs"):
        self.api = KISApi(mode=mode, market_type="domestic")
        self.dry_run = dry_run
        self.log_path = Path(log_dir) / "v4_trades.jsonl"

    @staticmethod
    def norm(ticker: str) -> str:
        """KRX 종목코드 6자리 정규화 (suffix/non-digit 제거, leading-zero 보존).

        KIS FID_INPUT_ISCD는 6자리 숫자만 허용 — .KS/.KQ suffix 전달 시 시세 0 반환.
        """
        digits = re.sub(r"\D", "", str(ticker))
        return digits.zfill(6)[-6:] if digits else str(ticker)

    def get_price_krw(self, ticker: str) -> float:
        try:
            return float(self.api.get_domestic_price(self.norm(ticker))["price"])
        except Exception as e:
            logger.warning(f"price fetch 실패 {ticker}: {e}")
            return 0.0

    def get_balance(self) -> dict:
        """{cash, total_eval, positions:[{ticker,qty,avg_price,current_price,...}]}."""
        return self.api.get_domestic_balance()

    def get_positions(self) -> dict[str, int]:
        return {self.norm(p["ticker"]): int(p["qty"]) for p in self.get_balance()["positions"]}

    def buy(self, ticker: str, amount_krw: float) -> dict | None:
        """금액 기반 매수 (KIS 시세로 qty 환산). 독립 사용/smoke 용 — executor는 buy_qty 사용."""
        t = self.norm(ticker)
        px = self.get_price_krw(t)
        if px <= 0:
            logger.error(f"BUY 불가 {t}: price=0")
            return None
        qty = int(amount_krw // px)
        if qty < 1:
            logger.warning(f"BUY 불가 {t}: amount {amount_krw:,.0f} < 1주 ({px:,.0f})")
            return None
        return self._order(t, "buy", qty, px, qty * px)

    def buy_qty(self, ticker: str, qty: int, ref_price: float = 0.0) -> dict | None:
        """수량 기반 시장가 매수. sizing은 호출자(executor)가 패널 종가로 — KIS 시세 불요.
        ref_price는 로그/금액 추정용(시장가라 실제 체결가는 별도)."""
        t = self.norm(ticker)
        qty = int(qty)
        if qty < 1:
            return None
        return self._order(t, "buy", qty, ref_price, qty * ref_price)

    def sell(self, ticker: str, qty: int, ref_price: float | None = None) -> dict | None:
        """수량 기반 시장가 매도. ref_price 주면 그 값 사용(KIS 시세 불요), None이면 조회."""
        t = self.norm(ticker)
        qty = int(qty)
        if qty < 1:
            return None
        px = ref_price if ref_price is not None else self.get_price_krw(t)
        return self._order(t, "sell", qty, px, qty * px)

    def _order(self, ticker: str, side: str, qty: int, px: float, amount: float) -> dict | None:
        if self.dry_run:
            res = {"order_no": "DRYRUN", "ticker": ticker, "side": side, "qty": qty,
                   "price": px, "dry_run": True}
        else:
            res = self._order_with_retry(ticker, side, qty, px)
            if res is None:
                return None
        self._log({**res, "amount_krw": amount, "time": datetime.now().isoformat(timespec="seconds")})
        logger.info(f"{'(dry) ' if self.dry_run else ''}{side.upper()} {ticker} x{qty} "
                    f"@~{px:,.0f} = {amount:,.0f} KRW")
        return res

    def _order_with_retry(self, ticker: str, side: str, qty: int, px: float) -> dict | None:
        """시장가 주문 1건. 일시적 에러는 MAX_ORDER_RETRIES 까지 재시도, 비일시적은 즉시 실패.

        2026-06-15 회귀: KIS sandbox 500/RemoteDisconnected 로 3종목 거부 → 재시도 없이
        누락. 500/연결끊김은 주문 미접수가 거의 확실(KIS가 접수 전 에러) → 재시도 안전.
        """
        for attempt in range(1, self.MAX_ORDER_RETRIES + 1):
            try:
                res = self.api.order_domestic(ticker, side, qty, order_type="01")  # 시장가
                res["price"] = px
                if attempt > 1:
                    logger.info(f"{side.upper()} {ticker} x{qty} 재시도 {attempt}회차 성공")
                return res
            except Exception as e:
                if _is_transient(e) and attempt < self.MAX_ORDER_RETRIES:
                    logger.warning(f"{side.upper()} {ticker} x{qty} 일시 실패 "
                                   f"(attempt {attempt}/{self.MAX_ORDER_RETRIES}): {e} — 재시도")
                    if self.RETRY_WAIT_S:
                        time.sleep(self.RETRY_WAIT_S * attempt)
                    continue
                logger.error(f"{side.upper()} 주문 실패 {ticker} x{qty} "
                             f"(attempt {attempt}, transient={_is_transient(e)}): {e}")
                return None
        return None

    def _log(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
