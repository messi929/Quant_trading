"""V4 reconcile executor — engine.target_book() 목표를 KIS 잔고와 맞춰 주문.

엔진(순수 함수)이 ticker→weight 목표를 내면, 현재 KIS 잔고와 비교해 청산/신규/조정
주문을 생성. broker 는 KisKoreaBroker 인터페이스(get_balance/get_price_krw/buy/sell/norm)
를 따르는 어떤 것이든 됨(테스트는 mock).

원칙:
  - long-only, no-margin: 총 목표 weight > 1.0 (vol-target cap1.5 leverage 구간)이면
    현금 계좌라 1.0으로 clamp (paper 는 무레버리지 variant 추적, leverage 기여는 미미).
  - target 에 없는 보유 종목 전량 청산 → target 별 desired qty 로 조정.
  - dust(min_order_krw 미만 조정) skip — 불필요 회전 비용 방지.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class RebalancePlan:
    sells: tuple[dict, ...]
    buys: tuple[dict, ...]
    clamped: bool
    total_eval: float


def rebalance(broker, targets: dict[str, float], *, min_order_krw: float = 500_000,
              max_total_weight: float = 1.0, execute: bool = True) -> RebalancePlan:
    """targets(ticker→weight) → KIS 잔고 reconcile. execute=False면 계획만(미주문)."""
    bal = broker.get_balance()
    total = float(bal.get("total_eval") or 0.0)
    if total <= 0:
        total = float(bal.get("cash") or 0.0)
    current = {broker.norm(p["ticker"]): int(p["qty"]) for p in bal.get("positions", [])}

    # no-margin clamp
    tw = sum(targets.values())
    clamped = tw > max_total_weight + 1e-9
    if clamped and tw > 0:
        scale = max_total_weight / tw
        targets = {t: w * scale for t, w in targets.items()}
        logger.info(f"target weight {tw:.2f} > {max_total_weight} → no-margin clamp ×{scale:.3f}")
    tgt = {broker.norm(t): w for t, w in targets.items()}

    sells, buys = [], []

    # 1. target 에 없는 보유 종목 전량 청산
    for tk, qty in current.items():
        if tk not in tgt and qty > 0:
            o = {"ticker": tk, "side": "sell", "qty": qty, "reason": "exit"}
            if execute:
                r = broker.sell(tk, qty)
                if r is None:
                    continue
                o.update(r)
            sells.append(o)

    # 2. target 별 desired qty 조정
    for tk, w in tgt.items():
        px = broker.get_price_krw(tk)
        if px <= 0:
            continue
        desired = int((w * total) // px)
        cur = current.get(tk, 0)
        diff = desired - cur
        if abs(diff) * px < min_order_krw:        # dust skip
            continue
        if diff > 0:
            o = {"ticker": tk, "side": "buy", "qty": diff, "reason": "adjust" if cur else "new"}
            if execute:
                r = broker.buy(tk, diff * px)
                if r is None:
                    continue
                o.update(r)
            buys.append(o)
        else:
            o = {"ticker": tk, "side": "sell", "qty": -diff, "reason": "trim"}
            if execute:
                r = broker.sell(tk, -diff)
                if r is None:
                    continue
                o.update(r)
            sells.append(o)

    logger.info(f"rebalance: total={total:,.0f} KRW, sells={len(sells)}, buys={len(buys)}"
                f"{' [clamped]' if clamped else ''}")
    return RebalancePlan(tuple(sells), tuple(buys), clamped, total)
