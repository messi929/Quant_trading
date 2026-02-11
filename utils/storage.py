"""Data storage utilities with HDF5, Parquet, and memory-mapped support."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class StorageManager:
    """Manages data persistence with multiple backend support."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_parquet(
        self,
        df: pd.DataFrame,
        name: str,
        subdir: str = "processed",
    ) -> Path:
        """Save DataFrame as Parquet (columnar, compressed)."""
        path = self.base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{name}.parquet"
        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        size_mb = filepath.stat().st_size / 1e6
        logger.info(f"Saved {filepath} ({size_mb:.1f} MB, {len(df)} rows)")
        return filepath

    def load_parquet(
        self,
        name: str,
        subdir: str = "processed",
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Load Parquet file with optional column selection."""
        filepath = self.base_dir / subdir / f"{name}.parquet"
        df = pd.read_parquet(filepath, columns=columns)
        logger.info(f"Loaded {filepath} ({len(df)} rows)")
        return df

    def save_numpy(
        self,
        arr: np.ndarray,
        name: str,
        subdir: str = "arrays",
        mmap: bool = False,
    ) -> Path:
        """Save numpy array, optionally as memory-mapped file."""
        path = self.base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{name}.npy"
        if mmap:
            fp = np.lib.format.open_memmap(
                str(filepath), mode="w+", dtype=arr.dtype, shape=arr.shape
            )
            fp[:] = arr[:]
            del fp
        else:
            np.save(filepath, arr)
        size_mb = filepath.stat().st_size / 1e6
        logger.info(f"Saved {filepath} ({size_mb:.1f} MB, shape={arr.shape})")
        return filepath

    def load_numpy(
        self,
        name: str,
        subdir: str = "arrays",
        mmap_mode: Optional[str] = "r",
    ) -> np.ndarray:
        """Load numpy array, optionally memory-mapped for large files."""
        filepath = self.base_dir / subdir / f"{name}.npy"
        arr = np.load(filepath, mmap_mode=mmap_mode)
        logger.info(f"Loaded {filepath} (shape={arr.shape})")
        return arr

    def save_model_checkpoint(
        self,
        state_dict: dict,
        name: str,
        epoch: int,
        metrics: Optional[dict] = None,
    ) -> Path:
        """Save model checkpoint with metadata."""
        import torch

        path = self.base_dir / ".." / "saved_models"
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{name}_epoch{epoch}.pt"
        checkpoint = {
            "epoch": epoch,
            "state_dict": state_dict,
            "metrics": metrics or {},
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        return filepath

    def load_model_checkpoint(self, name: str, epoch: Optional[int] = None) -> dict:
        """Load model checkpoint. If epoch is None, load latest."""
        import torch

        path = self.base_dir / ".." / "saved_models"
        if epoch is not None:
            filepath = path / f"{name}_epoch{epoch}.pt"
        else:
            checkpoints = sorted(path.glob(f"{name}_epoch*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found for {name}")
            filepath = checkpoints[-1]
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        logger.info(f"Loaded checkpoint: {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint

    def list_files(self, subdir: str = "", pattern: str = "*") -> list[Path]:
        """List files in a subdirectory matching a pattern."""
        path = self.base_dir / subdir
        return sorted(path.glob(pattern)) if path.exists() else []
