"""CLI: Run V3 live trading session or daemon."""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.config.schema import load_config
from v3.pipeline.live_pipeline import LivePipeline
from v3.utils.logger import setup_logger
from loguru import logger


def run_once(cfg):
    """Run a single trading session."""
    pipeline = LivePipeline(cfg)
    result = pipeline.run_session()

    print("\n" + "=" * 50)
    print("V3 Trading Session Summary")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    return result


def run_daemon(cfg):
    """Run as daemon with scheduled sessions."""
    pipeline = LivePipeline(cfg)
    kr_schedule = cfg.schedule.kr
    us_schedule = cfg.schedule.us

    logger.info("V3 Trading Daemon started")
    logger.info(f"KR: collect={kr_schedule.data_collect}, "
                f"inference={kr_schedule.inference}, "
                f"execute={kr_schedule.execute}")
    logger.info(f"US: inference={us_schedule.inference}, "
                f"execute={us_schedule.execute}")

    last_kr_date = ""
    last_us_date = ""

    while True:
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        date_str = now.strftime("%Y-%m-%d")

        # KR session: execute at configured time
        if time_str == kr_schedule.execute and date_str != last_kr_date:
            if now.weekday() < 5:  # Weekdays only
                logger.info("=== KR Session ===")
                try:
                    result = pipeline.run_session()
                    last_kr_date = date_str
                except Exception as e:
                    logger.error(f"KR session failed: {e}")

        # US session
        if time_str == us_schedule.execute and date_str != last_us_date:
            if now.weekday() < 5:
                logger.info("=== US Session ===")
                try:
                    result = pipeline.run_session()
                    last_us_date = date_str
                except Exception as e:
                    logger.error(f"US session failed: {e}")

        # Monitor: every N minutes
        monitor_interval = kr_schedule.monitor_interval_min * 60
        if hasattr(pipeline, '_last_monitor'):
            if (now.timestamp() - pipeline._last_monitor) >= monitor_interval:
                try:
                    exits = pipeline.monitor()
                    if exits:
                        logger.info(f"Monitor exits: {len(exits)}")
                    pipeline._last_monitor = now.timestamp()
                except Exception as e:
                    logger.error(f"Monitor failed: {e}")
        else:
            pipeline._last_monitor = now.timestamp()

        time.sleep(10)  # Check every 10 seconds


def main():
    parser = argparse.ArgumentParser(description="V3 Live Trading")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=["once", "daemon"], default="once",
                        help="Run once or as daemon")
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)

    if args.mode == "once":
        run_once(cfg)
    else:
        run_daemon(cfg)


if __name__ == "__main__":
    main()
