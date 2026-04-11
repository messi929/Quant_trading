"""PyTorch Dataset for volatility expansion prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class VolDataset(Dataset):
    """Sequence dataset for volatility expansion prediction.

    Each sample: (features[t-seq_len:t], vol_target[t])
    - features: (seq_len, n_features) normalized float tensor
    - target: scalar float — vol expansion ratio
    - confidence_target: 1 if prediction direction correct (for confidence head)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "vol_target",
        seq_len: int = 60,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.samples: list[dict] = []

        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        """Build (sequence, target) pairs per ticker."""
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            # Extract feature matrix and targets
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_col].values.astype(np.float32)
            dates = group["date"].values

            # Create sliding windows
            for i in range(self.seq_len, len(group)):
                if np.isnan(targets[i]):
                    continue

                feat_window = features[i - self.seq_len : i]

                # Skip if too many NaNs in window
                nan_ratio = np.isnan(feat_window).mean()
                if nan_ratio > 0.3:
                    continue

                # Fill remaining NaNs with 0 (after z-score, 0 = mean)
                feat_window = np.nan_to_num(feat_window, nan=0.0)

                self.samples.append({
                    "features": feat_window,
                    "target": targets[i],
                    "ticker": ticker,
                    "date": str(dates[i]),
                })

        logger.info(f"VolDataset: {len(self.samples)} samples, "
                     f"{len(self.feature_cols)} features, seq_len={self.seq_len}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
        }


def create_dataloaders(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "vol_target",
    seq_len: int = 60,
    batch_size: int = 256,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """Create train/val/test dataloaders with temporal split.

    Returns:
        (train_loader, val_loader, test_loader, test_df)
    """
    # Temporal split — NO shuffling across time
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    logger.info(f"Data split: train={len(train_df)} ({len(train_dates)} days), "
                 f"val={len(val_df)} ({len(val_dates)} days), "
                 f"test={len(test_df)} ({len(test_dates)} days)")

    train_ds = VolDataset(train_df, feature_cols, target_col, seq_len)
    val_ds = VolDataset(val_df, feature_cols, target_col, seq_len)
    test_ds = VolDataset(test_df, feature_cols, target_col, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, test_df
