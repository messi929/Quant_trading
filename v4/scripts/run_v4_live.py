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
from v4.live.state import TrancheLiveState
from v4.live.tranche_runner import run_tranche_session


def session(broker, cfg, universe_size: int, execute: bool) -> None:
    close, dvol, index = build_live_panel(cfg, universe_size=universe_size)
    state = TrancheLiveState.load(n_tranches=cfg.n_tranches)
    res = run_tranche_session(broker, close, dvol, index, state, cfg, execute=execute)
    state.save()
    active = sum(s.last_rebalance_date is not None for s in state.tranches)
    if res.note == "rebalanced":
        logger.info(f"세션 완료: 트렌치 {list(res.rebalanced_tranches)} 리밸런스 "
                    f"(활성 {active}/{state.n_tranches}) 결합노출={res.gross_exposure:.2f} "
                    f"종목={res.n_positions} sells={len(res.plan.sells)} "
                    f"buys={len(res.plan.buys)} fails={len(res.plan.failures)}")
    elif res.note == "exec_failed":
        logger.error(f"세션 실패: 전 주문 실행 실패 ({len(res.plan.failures)}건) — "
                     f"트렌치 state 미전진, 다음 세션 재시도. 계좌 리셋/예수금 점검 필요.")
    else:
        logger.info(f"세션 완료: hold (due 트렌치 없음, 활성 {active}/{state.n_tranches})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["once", "daemon"], default="once")
    ap.add_argument("--dry-run", action="store_true", help="주문 API 미호출 (의도만 로그)")
    ap.add_argument("--no-execute", action="store_true", help="reconcile 계획만, 주문 안 함")
    ap.add_argument("--universe-size", type=int, default=400)
    args = ap.parse_args()

    # loguru 기본 sink 는 stderr → systemd StandardError(quant-v4-error.log)로만 가고
    # 문서상 모니터링 파일 StandardOutput(quant-v4.log)은 빈 채였다(6/1~ 가시성 공백).
    # stdout 으로 보내 quant-v4.log 에 세션 로그가 남게 한다.
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                      "{name}:{function}:{line} - {message}")

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
