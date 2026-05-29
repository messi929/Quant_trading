"""KIS 국내 연동 smoke 테스트 — 토큰/가격/잔고 (read-only, 주문 없음).

Stage2 연동 확인용. 토큰 발급 + 잔고 + 샘플 시세. 주문은 하지 않음(dry_run).
시세/토큰은 장외에도 동작, 잔고는 항상 동작. 실주문 체결만 KR 장중.

Usage:
    PYTHONPATH=. python v4/scripts/kis_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from v4.execution.kis_broker import KisKoreaBroker


def main() -> int:
    logger.info("KIS 국내 sandbox 연동 확인 (read-only)...")
    broker = KisKoreaBroker(mode="sandbox", dry_run=True)

    try:
        bal = broker.get_balance()
    except Exception as e:
        logger.error(f"잔고 조회 실패 (토큰/키 확인): {e}")
        return 1
    logger.info(f"계좌: 예수금={bal['cash']:,.0f} 총평가={bal['total_eval']:,.0f} "
                f"보유={len(bal['positions'])}종목")

    for t, name in [("005930", "삼성전자"), ("035720", "카카오"), ("247540", "에코프로비엠")]:
        px = broker.get_price_krw(t)
        logger.info(f"  {t} {name}: {px:,.0f} KRW")

    logger.info("연동 OK." if bal else "연동 확인 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
