"""V2 Trading Daemon entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scheduler.runner import TradingRunner

if __name__ == "__main__":
    TradingRunner().run_daemon()
