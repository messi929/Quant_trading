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
    failures: tuple[dict, ...] = ()      # 시도했으나 broker None 반환 (매매불가/자금부족/장외)


def rebalance(broker, targets: dict[str, float], prices: dict[str, float], *,
              min_order_krw: float = 500_000, max_total_weight: float = 1.0,
              execute: bool = True) -> RebalancePlan:
    """targets(ticker→weight) → KIS 잔고 reconcile. sizing은 prices(패널 종가)로 —
    KIS 실시간 시세 불요(500에러 회피) + backtest parity(종가 sizing). execute=False면 계획만."""
    bal = broker.get_balance()
    total = float(bal.get("total_eval") or 0.0)
    if total <= 0:
        total = float(bal.get("cash") or 0.0)
    current = {broker.norm(p["ticker"]): int(p["qty"]) for p in bal.get("positions", [])}
    px_of = {broker.norm(t): p for t, p in prices.items()}

    # no-margin clamp
    tw = sum(targets.values())
    clamped = tw > max_total_weight + 1e-9
    if clamped and tw > 0:
        scale = max_total_weight / tw
        targets = {t: w * scale for t, w in targets.items()}
        logger.info(f"target weight {tw:.2f} > {max_total_weight} → no-margin clamp ×{scale:.3f}")
    tgt = {broker.norm(t): w for t, w in targets.items()}

    sells, buys, failures = [], [], []

    # 1. target 에 없는 보유 종목 전량 청산 (ref_price=패널 종가, 없으면 broker 조회)
    for tk, qty in current.items():
        if tk not in tgt and qty > 0:
            o = {"ticker": tk, "side": "sell", "qty": qty, "reason": "exit"}
            if execute:
                r = broker.sell(tk, qty, ref_price=px_of.get(tk))
                if r is None:
                    logger.error(f"order FAILED sell {tk} x{qty} (exit) — broker None "
                                 f"(매매불가/장외?)")
                    failures.append(o)
                    continue
                o.update(r)
            sells.append(o)

    # 2. target 별 desired qty 조정 — 패널 종가로 sizing
    for tk, w in tgt.items():
        px = px_of.get(tk)
        if px is None or px <= 0:
            logger.warning(f"sizing skip {tk}: 패널 종가 없음")
            continue
        desired = int((w * total) // px)
        cur = current.get(tk, 0)
        diff = desired - cur
        if abs(diff) * px < min_order_krw:        # dust skip
            continue
        if diff > 0:
            o = {"ticker": tk, "side": "buy", "qty": diff, "reason": "adjust" if cur else "new"}
            if execute:
                r = broker.buy_qty(tk, diff, ref_price=px)
                if r is None:
                    logger.error(f"order FAILED buy {tk} x{diff} ({o['reason']}) — broker "
                                 f"None (자금부족/매매불가?)")
                    failures.append(o)
                    continue
                o.update(r)
            buys.append(o)
        else:
            o = {"ticker": tk, "side": "sell", "qty": -diff, "reason": "trim"}
            if execute:
                r = broker.sell(tk, -diff, ref_price=px)
                if r is None:
                    logger.error(f"order FAILED sell {tk} x{-diff} (trim) — broker None")
                    failures.append(o)
                    continue
                o.update(r)
            sells.append(o)

    if failures:
        logger.error(f"rebalance: {len(failures)} order(s) FAILED — "
                     f"{[f['ticker'] for f in failures][:10]}")
    logger.info(f"rebalance: total={total:,.0f} KRW, sells={len(sells)}, buys={len(buys)}, "
                f"fails={len(failures)}{' [clamped]' if clamped else ''}")
    return RebalancePlan(tuple(sells), tuple(buys), clamped, total, tuple(failures))
