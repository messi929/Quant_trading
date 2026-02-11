"""Registry for discovered indicators with persistence and versioning."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class IndicatorRegistry:
    """Stores and manages discovered indicators across training sessions."""

    def __init__(self, save_dir: str = "saved_indicators"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.indicators: dict[str, dict] = {}

    def register(
        self,
        name: str,
        values: np.ndarray,
        metadata: dict,
        evaluation: dict,
        version: Optional[str] = None,
    ) -> None:
        """Register a discovered indicator.

        Args:
            name: Indicator name (e.g., "alpha_0_vol")
            values: Indicator values array
            metadata: Interpretation and correlation info
            evaluation: Quality evaluation scores
            version: Version string (auto-generated if None)
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.indicators[name] = {
            "name": name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "n_samples": len(values),
            "stats": {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "skew": float(pd.Series(values).skew()),
                "kurtosis": float(pd.Series(values).kurtosis()),
            },
            "metadata": metadata,
            "evaluation": {k: float(v) for k, v in evaluation.items()},
        }

        logger.info(
            f"Registered indicator: {name} (v{version}, "
            f"score={evaluation.get('composite_score', 0):.4f})"
        )

    def save(self) -> Path:
        """Save all registered indicators to disk."""
        filepath = self.save_dir / "indicator_registry.json"

        # Convert to JSON-serializable format
        data = {}
        for name, info in self.indicators.items():
            data[name] = {
                k: v for k, v in info.items()
                if k != "values"  # Don't save raw values in JSON
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} indicators to {filepath}")
        return filepath

    def load(self, filepath: Optional[str] = None) -> None:
        """Load indicator registry from disk."""
        filepath = Path(filepath or self.save_dir / "indicator_registry.json")
        if not filepath.exists():
            logger.warning(f"Registry file not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            self.indicators = json.load(f)

        logger.info(f"Loaded {len(self.indicators)} indicators from {filepath}")

    def get_top_indicators(
        self,
        n: int = 10,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Get top-N indicators by composite score."""
        scored = [
            (name, info)
            for name, info in self.indicators.items()
            if info.get("evaluation", {}).get("composite_score", 0) >= min_score
        ]
        scored.sort(
            key=lambda x: x[1]["evaluation"]["composite_score"],
            reverse=True,
        )
        return [info for _, info in scored[:n]]

    def get_indicator(self, name: str) -> Optional[dict]:
        """Get a specific indicator's info."""
        return self.indicators.get(name)

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all registered indicators."""
        if not self.indicators:
            return pd.DataFrame()

        records = []
        for name, info in self.indicators.items():
            record = {
                "name": name,
                "version": info.get("version", ""),
            }
            record.update(info.get("stats", {}))
            record.update(info.get("evaluation", {}))
            records.append(record)

        return pd.DataFrame(records).sort_values(
            "composite_score", ascending=False
        )
