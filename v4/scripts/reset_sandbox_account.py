"""V4 sandbox 국내계좌 리셋 — V1/V2 잔여 전량 청산 → 현금화 (paper 1억 시작점).

V4 KOSDAQ paper 가동 전제조건 #2. sandbox 국내계좌(CANO 50169471)에 V1/V2가
남긴 잔여 종목을 전량 시장가 매도해 현금 상태로 되돌린다.

안전장치:
- 기본 **dry-run** (주문 미발생, 잔고/매도계획만 출력). 실주문은 명시적 --execute.
- 실체결은 **KR 장중에만** (장외 --execute 시 KIS가 주문 거부 → 로그만).
- 토큰/잔고 조회는 장외에도 동작 → 주말에 dry-run으로 사전 점검 가능.

Usage:
    PYTHONPATH=. python v4/scripts/reset_sandbox_account.py            # dry-run (점검)
    PYTHONPATH=. python v4/scripts/reset_sandbox_account.py --execute  # 실청산 (장중)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from v4.execution.kis_broker import KisKoreaBroker


def reset(broker: KisKoreaBroker, execute: bool) -> int:
    bal = broker.get_balance()
    positions = bal.get("positions", [])
    cash = float(bal.get("cash", 0) or 0)
    total = float(bal.get("total_eval", 0) or 0)
    logger.info(f"현재 잔고: cash={cash:,.0f}  total_eval={total:,.0f}  "
                f"보유종목={len(positions)}")
    for p in positions:
        logger.info(f"  {p['ticker']:>8} x{int(p['qty']):>6}  "
                    f"avg={float(p.get('avg_price', 0) or 0):>10,.0f}  "
                    f"cur={float(p.get('current_price', 0) or 0):>10,.0f}")

    sellable = [p for p in positions if int(p.get("qty", 0)) >= 1]
    if not sellable:
        logger.info("청산할 포지션 없음 — 이미 현금 상태. 리셋 불필요.")
        return 0

    logger.info(f"{'>>> 실행(실주문)' if execute else '>>> DRY-RUN(주문 미발생)'}: "
                f"{len(sellable)}종목 전량 시장가 매도")
    for p in sellable:
        broker.sell(p["ticker"], int(p["qty"]),
                    ref_price=float(p.get("current_price") or 0))

    if execute:
        after = broker.get_balance()
        logger.info(f"리셋 후 잔고: cash={float(after.get('cash', 0) or 0):,.0f}  "
                    f"보유종목={len(after.get('positions', []))}")
        remaining = [p for p in after.get("positions", []) if int(p.get("qty", 0)) >= 1]
        if remaining:
            logger.warning(f"미청산 {len(remaining)}종목 남음 (장외/거래정지/체결지연?) — "
                           f"장중 재실행 필요")
        else:
            logger.info("✓ 전량 청산 완료 — 현금 리셋됨. V4 paper 가동 준비.")
    else:
        logger.info("dry-run 완료. 실청산은 KR 장중 `--execute`로 실행.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="실주문 청산 (KR 장중에만 체결). 미지정 시 dry-run")
    args = ap.parse_args()
    broker = KisKoreaBroker(mode="sandbox", dry_run=not args.execute)
    return reset(broker, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
