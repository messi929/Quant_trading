"""V4 Korea 엔진 live 실행 — KIS 국내 paper.

매 세션: 현재 KOSDAQ 패널 build → runner.run_session (rebalance-or-hold) →
KIS 국내 주문 → state 저장. momentum 20일 보유라 rebalance일에만 매매.

권장 운영: KR 종가 후(예: 15:40 KST) systemd timer가 `--mode once` 1일 1회 호출.

Usage:
    PYTHONPATH=. python v4/scripts/run_v4_live.py --mode once --dry-run
    PYTHONPATH=. python v4/scripts/run_v4_live.py --mode once            # 실주문(sandbox)
    PYTHONPATH=. python v4/scripts/run_v4_live.py --mode once --no-execute --universe-size 40  # plan만
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from v4.config import KoreaConfig
from v4.execution.kis_broker import KisKoreaBroker
from v4.live.data_live import build_live_panel
from v4.live.runner import run_session
from v4.live.state import LiveState


def session(broker, cfg, universe_size: int, execute: bool) -> None:
    close, dvol, index = build_live_panel(cfg, universe_size=universe_size)
    state = LiveState.load()
    res = run_session(broker, close, dvol, index, state, cfg, execute=execute)
    state.save()
    if res.rebalanced:
        logger.info(f"세션 완료: regime={'ON' if res.regime_on else 'OFF'} "
                    f"exposure={res.exposure:.2f} picks={res.n_picks} "
                    f"sells={len(res.plan.sells)} buys={len(res.plan.buys)} "
                    f"fails={len(res.plan.failures)}")
    elif res.note == "exec_failed":
        logger.error(f"세션 실패: 전 주문 실행 실패 ({len(res.plan.failures)}건) — "
                     f"state 미전진, 다음 세션 재시도. 계좌 리셋/예수금 점검 필요.")
    else:
        logger.info("세션 완료: hold (rebalance일 아님)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["once", "daemon"], default="once")
    ap.add_argument("--dry-run", action="store_true", help="주문 API 미호출 (의도만 로그)")
    ap.add_argument("--no-execute", action="store_true", help="reconcile 계획만, 주문 안 함")
    ap.add_argument("--universe-size", type=int, default=400)
    args = ap.parse_args()

    cfg = KoreaConfig()
    broker = KisKoreaBroker(mode="sandbox", dry_run=args.dry_run)
    execute = not args.no_execute

    if args.mode == "once":
        session(broker, cfg, args.universe_size, execute)
        return 0

    logger.info("daemon: 1일 1회 세션 (KR 거래일). Ctrl-C 종료.")
    while True:
        try:
            session(broker, cfg, args.universe_size, execute)
        except Exception as e:
            logger.error(f"세션 오류: {e}")
        time.sleep(24 * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
