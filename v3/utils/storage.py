"""Model checkpoint and data storage management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from loguru import logger


class StorageManager:
    """Manages model checkpoints and data persistence."""

    def __init__(self, base_dir: str = "v3"):
        self.base_dir = Path(base_dir)

    def save_parquet(self, df: pd.DataFrame, name: str, subdir: str = "data/processed") -> Path:
        path = self.base_dir / subdir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy")
        logger.info(f"Saved parquet: {path} ({len(df)} rows)")
        return path

    def load_parquet(self, name: str, subdir: str = "data/processed", columns: list[str] | None = None) -> pd.DataFrame:
        path = self.base_dir / subdir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        df = pd.read_parquet(path, columns=columns)
        logger.info(f"Loaded parquet: {path} ({len(df)} rows)")
        return df

    def save_checkpoint(self, state_dict: dict, name: str, epoch: int, metrics: dict | None = None) -> Path:
        path = self.base_dir / "saved_models" / f"{name}_epoch{epoch:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": state_dict, "epoch": epoch, "metrics": metrics or {}}
        torch.save(payload, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, name: str, epoch: int | None = None) -> dict:
        model_dir = self.base_dir / "saved_models"
        if epoch is not None:
            path = model_dir / f"{name}_epoch{epoch:03d}.pt"
        else:
            candidates = sorted(model_dir.glob(f"{name}_epoch*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoints found for {name}")
            path = candidates[-1]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Checkpoint loaded: {path} (epoch {payload.get('epoch', '?')})")
        return payload

    def save_json(self, data: Any, name: str, subdir: str = "saved_models") -> Path:
        path = self.base_dir / subdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def load_json(self, name: str, subdir: str = "saved_models") -> Any:
        path = self.base_dir / subdir / name
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
