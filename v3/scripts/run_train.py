"""CLI: Run V3 training pipeline."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.config.schema import load_config
from v3.pipeline.train_pipeline import TrainPipeline
from v3.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="V3 Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    setup_logger()

    cfg = load_config(args.config)
    pipeline = TrainPipeline(cfg)
    result = pipeline.run()

    print("\n" + "=" * 50)
    print("V3 Training Summary")
    print("=" * 50)
    val_pass = result["best_metrics"].get("vol_rank_ic", 0) >= 0.20
    test_pass = result["test_metrics"].get("gate_pass", False)
    print(f"  Val Gate:  {'PASS' if val_pass else 'FAIL'}")
    print(f"  Test Gate: {'PASS' if test_pass else 'FAIL'}")
    print(f"  Overall:   {'PASS' if result['gate_pass'] else 'FAIL'}")

    for k, v in result["test_metrics"].items():
        if isinstance(v, float):
            print(f"  Test {k}: {v:.4f}")

    print("=" * 50)

    if not result["gate_pass"]:
        print("\n⚠ GATE FAILED — do NOT proceed to backtest.")
        print("  Improve features or model architecture first.")


if __name__ == "__main__":
    main()
