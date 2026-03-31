"""PyTorch Dataset and DataLoader for sector-based time series.

v2: Added FeatureNormalizer integration in create_dataloaders().
    Training-set statistics are used for all splits (no look-ahead).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from data.normalizer import FeatureNormalizer


class MarketSequenceDataset(Dataset):
    """Time-series dataset that creates sequences for each ticker.

    Each sample is a (sequence, target, sector_id, ticker) tuple where:
    - sequence: (seq_len, n_features) tensor of historical features
    - target: future return over prediction_horizon
    - sector_id: integer sector index
    - ticker: string ticker symbol
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_length: int = 60,
        prediction_horizon: int = 5,
        sector_map: Optional[dict[str, int]] = None,
        cross_sectional_target: bool = True,
    ):
        """
        Args:
            df: DataFrame with features, must have 'ticker', 'date', 'close', 'sector'
            feature_cols: List of feature column names
            seq_length: Lookback window length
            prediction_horizon: Days ahead for target return
            sector_map: Mapping from sector name to integer ID
            cross_sectional_target: If True, target is relative rank instead of
                absolute return. This makes the model predict which stocks will
                outperform peers, not whether the market goes up.
        """
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.feature_cols = feature_cols
        self.sector_map = sector_map or {}
        self.cross_sectional_target = cross_sectional_target

        # Pre-compute cross-sectional targets if enabled
        self._date_market_return = {}
        if cross_sectional_target:
            # Compute per-date market average return for relative targeting
            df = df.sort_values(["ticker", "date"]).copy()
            df["_fwd_return"] = df.groupby("ticker")["close"].transform(
                lambda x: x.shift(-prediction_horizon) / x - 1
            )
            # Per-date mean forward return (equal-weight market return)
            date_avg = df.groupby("date")["_fwd_return"].mean()
            self._date_market_return = date_avg.to_dict()
            df = df.drop(columns=["_fwd_return"])

        # Build index of valid (ticker, start_idx) pairs
        self.samples = []
        self.data = {}  # ticker -> (features_array, close_array, sector_id, dates_array)

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[feature_cols].values.astype(np.float32)
            # Replace any residual NaN/inf (safety net after normalization)
            features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
            closes = group["close"].values.astype(np.float32)
            dates = group["date"].values  # for cross-sectional lookup
            sector = group["sector"].iloc[0] if "sector" in group.columns else "unknown"
            sector_id = self.sector_map.get(sector, 0)

            self.data[ticker] = (features, closes, sector_id, dates)

            # Valid start indices: need seq_length + prediction_horizon rows
            n_valid = len(group) - seq_length - prediction_horizon
            for i in range(max(0, n_valid)):
                self.samples.append((ticker, i))

        logger.info(
            f"Dataset: {len(self.samples)} samples from "
            f"{len(self.data)} tickers, "
            f"seq_len={seq_length}, horizon={prediction_horizon}"
            f"{', cross-sectional target' if cross_sectional_target else ''}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        ticker, start = self.samples[idx]
        features, closes, sector_id, dates = self.data[ticker]

        # Extract sequence
        end = start + self.seq_length
        seq = features[start:end]

        # Target: future log return
        current_close = closes[end - 1]
        future_close = closes[end - 1 + self.prediction_horizon]
        abs_return = np.log(future_close / current_close) if current_close > 0 else 0.0

        if self.cross_sectional_target:
            # Relative return: stock return minus market average return
            # This makes the model predict outperformance, not direction
            current_date = dates[end - 1]
            market_return = self._date_market_return.get(current_date, 0.0)
            if np.isnan(market_return):
                market_return = 0.0
            # Convert market return to log scale for consistency
            market_log_return = np.log(1 + market_return) if market_return > -1 else 0.0
            target = abs_return - market_log_return
        else:
            target = abs_return

        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(sector_id, dtype=torch.long),
        )


class SectorBatchSampler:
    """Samples batches ensuring sector diversity within each batch."""

    def __init__(
        self,
        dataset: MarketSequenceDataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by sector
        self.sector_indices: dict[int, list[int]] = {}
        for idx, (ticker, _) in enumerate(dataset.samples):
            _, _, sector_id = dataset.data[ticker]
            self.sector_indices.setdefault(sector_id, []).append(idx)

    def __iter__(self):
        # Flatten and optionally shuffle
        all_indices = []
        for indices in self.sector_indices.values():
            if self.shuffle:
                perm = np.random.permutation(len(indices))
                all_indices.extend([indices[i] for i in perm])
            else:
                all_indices.extend(indices)

        if self.shuffle:
            np.random.shuffle(all_indices)

        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i : i + self.batch_size]

    def __len__(self):
        total = sum(len(v) for v in self.sector_indices.values())
        return (total + self.batch_size - 1) // self.batch_size


def create_dataloaders(
    df: pd.DataFrame,
    feature_cols: list[str],
    sector_map: dict[str, int],
    seq_length: int = 60,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    cross_sectional_target: bool = True,
    normalize: bool = True,
    normalizer_save_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, FeatureNormalizer | None]:
    """Create train/val/test DataLoaders with time-based splitting.

    Split is done temporally: oldest data for train, middle for val,
    newest for test to prevent look-ahead bias.

    v2: Feature normalization using training-set statistics.

    Args:
        normalize: If True, z-score normalize features using train stats.
        normalizer_save_path: If set, save normalizer stats to this path.

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
        normalizer is None if normalize=False.
    """
    df = df.sort_values("date")
    dates = df["date"].unique()
    n_dates = len(dates)

    train_end = dates[int(n_dates * train_ratio)]
    val_end = dates[int(n_dates * (train_ratio + val_ratio))]

    train_df = df[df["date"] < train_end].copy()
    val_df = df[(df["date"] >= train_end) & (df["date"] < val_end)].copy()
    test_df = df[df["date"] >= val_end].copy()

    logger.info(
        f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # v2: Feature normalization (train stats only)
    normalizer = None
    if normalize:
        normalizer = FeatureNormalizer()
        normalizer.fit(train_df, feature_cols)
        train_df = normalizer.transform(train_df, feature_cols)
        val_df = normalizer.transform(val_df, feature_cols)
        test_df = normalizer.transform(test_df, feature_cols)
        if normalizer_save_path:
            normalizer.save(normalizer_save_path)
        logger.info("Feature normalization applied (train-set statistics)")

    train_ds = MarketSequenceDataset(
        train_df, feature_cols, seq_length, prediction_horizon, sector_map,
        cross_sectional_target=cross_sectional_target,
    )
    val_ds = MarketSequenceDataset(
        val_df, feature_cols, seq_length, prediction_horizon, sector_map,
        cross_sectional_target=cross_sectional_target,
    )
    test_ds = MarketSequenceDataset(
        test_df, feature_cols, seq_length, prediction_horizon, sector_map,
        cross_sectional_target=cross_sectional_target,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, normalizer
