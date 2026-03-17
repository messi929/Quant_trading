"""
로또 번호 자동 예측 및 텔레그램 발송 모듈.
매주 수요일 10:00 daily_runner 데몬에서 호출됨.
"""

import sys
from pathlib import Path

# 로또 봇 경로 등록 (scheduler/../lotto)
LOTTO_PATH = Path(__file__).parent.parent / "lotto"
if str(LOTTO_PATH) not in sys.path:
    sys.path.insert(0, str(LOTTO_PATH))

from loguru import logger


def run():
    """매주 수요일 10시 호출 진입점."""
    logger.info("=== [LOTTO] 주간 로또 분석 시작 ===")
    try:
        from Auto_lotto_telegram import run_weekly_analysis
        run_weekly_analysis()
        logger.info("=== [LOTTO] 텔레그램 발송 완료 ===")
    except Exception as e:
        logger.error(f"[LOTTO] 실행 오류: {e}")
