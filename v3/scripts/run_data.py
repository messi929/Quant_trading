"""CLI: Run V3 data pipeline."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.config.schema import load_config
from v3.pipeline.data_pipeline import DataPipeline
from v3.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="V3 Data Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--incremental", type=int, default=None,
                        help="Incremental days (e.g., 10 for last 10 days)")
    parser.add_argument("--no-flow", action="store_true",
                        help="Skip flow data collection (faster)")
    args = parser.parse_args()

    setup_logger()

    cfg = load_config(args.config)
    pipeline = DataPipeline(cfg)
    result = pipeline.run(
        incremental_days=args.incremental,
        collect_flow=not args.no_flow,
    )

    # Print summary
    stats = result["stats"]
    if stats:
        print("\n" + "=" * 50)
        print("V3 Data Pipeline Summary")
        print("=" * 50)
        print(f"  Tickers:  {stats['n_tickers']}")
        print(f"  Features: {stats['n_features']}")
        print(f"  Samples:  {stats['valid_samples']}")
        print(f"  Date:     {stats['date_range']}")
        print(f"  Target μ: {stats['target_mean']:.4f}")
        print(f"  Target σ: {stats['target_std']:.4f}")
        print(f"  Flow:     {'Yes' if stats['has_flow'] else 'No'}")

        # Top feature ICs
        if "feature_ic" in stats:
            print("\n  Top Feature ICs (vs vol_target):")
            ic = stats["feature_ic"]
            top = sorted(ic.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for name, val in top:
                print(f"    {name:30s}  IC={val:+.4f}")

        print("=" * 50)


if __name__ == "__main__":
    main()
