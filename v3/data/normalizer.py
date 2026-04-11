"""Feature normalizer for z-score standardization.

Computes per-column mean/std on training data, applies to all splits.
Stats are saved alongside model checkpoints for inference consistency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class FeatureNormalizer:
    """Per-column z-score normalizer with train-set statistics."""

    def __init__(self, clip_value: float = 5.0):
        self.clip_value = clip_value
        self.mean_: dict[str, float] | None = None
        self.std_: dict[str, float] | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "FeatureNormalizer":
        self.mean_ = {}
        self.std_ = {}

        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not in DataFrame, skipping")
                continue
            values = df[col].dropna()
            self.mean_[col] = float(values.mean())
            std = float(values.std())
            self.std_[col] = std if std > 1e-8 else 1.0

        self._fitted = True
        logger.info(f"FeatureNormalizer fitted: {len(self.mean_)} columns")
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list[str], inplace: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer not fitted. Call fit() first.")

        if not inplace:
            df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                continue
            mean = self.mean_.get(col, 0.0)
            std = self.std_.get(col, 1.0)
            df[col] = ((df[col] - mean) / std).clip(-self.clip_value, self.clip_value)

        return df

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"clip_value": self.clip_value, "mean": self.mean_, "std": self.std_}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Normalizer stats saved: {path} ({len(self.mean_)} columns)")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer stats not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalizer = cls(clip_value=data.get("clip_value", 5.0))
        normalizer.mean_ = data["mean"]
        normalizer.std_ = data["std"]
        normalizer._fitted = True
        logger.info(f"Normalizer loaded: {path} ({len(normalizer.mean_)} columns)")
        return normalizer

    @property
    def is_fitted(self) -> bool:
        return self._fitted
