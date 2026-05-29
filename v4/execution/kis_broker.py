"""V4 KIS 국내 broker 래퍼 — broker/kis_api.py(KISApi domestic) 재사용.

KIS sandbox 모의투자 계좌(국내, ₩1억)에 실제 주문/잔고 연동. 계좌 상태는 KIS가
추적(V3 yfinance 시뮬과 달리) → 실자본 전환 시 mode="prod"만 변경.

dry_run=True: 주문 API 미호출, 의도만 로그 (live 가격으로 파이프라인 검증용).

주의: 실주문 체결은 KR 장중에만. 토큰/가격 조회는 장외에도 가능.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from broker.kis_api import KISApi


class KisKoreaBroker:
    """KISApi 국내 얇은 래퍼. price_krw / buy(amount) / sell(qty) / balance."""

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

    def sell(self, ticker: str, qty: int) -> dict | None:
        t = self.norm(ticker)
        qty = int(qty)
        if qty < 1:
            return None
        px = self.get_price_krw(t)
        return self._order(t, "sell", qty, px, qty * px)

    def _order(self, ticker: str, side: str, qty: int, px: float, amount: float) -> dict | None:
        if self.dry_run:
            res = {"order_no": "DRYRUN", "ticker": ticker, "side": side, "qty": qty,
                   "price": px, "dry_run": True}
        else:
            try:
                res = self.api.order_domestic(ticker, side, qty, order_type="01")  # 시장가
                res["price"] = px
            except Exception as e:
                logger.error(f"{side.upper()} 주문 실패 {ticker} x{qty}: {e}")
                return None
        self._log({**res, "amount_krw": amount, "time": datetime.now().isoformat(timespec="seconds")})
        logger.info(f"{'(dry) ' if self.dry_run else ''}{side.upper()} {ticker} x{qty} "
                    f"@~{px:,.0f} = {amount:,.0f} KRW")
        return res

    def _log(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
