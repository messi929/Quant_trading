"""Batch inference for VolTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from v3.model.vol_transformer import VolTransformer
from v3.data.normalizer import FeatureNormalizer
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


class VolInference:
    """Run VolTransformer inference on latest data."""

    def __init__(
        self,
        model: VolTransformer,
        normalizer: FeatureNormalizer,
        feature_cols: list[str],
        device_manager: DeviceManager,
        seq_len: int = 60,
    ):
        self.model = model.to(device_manager.device)
        self.model.eval()
        self.normalizer = normalizer
        self.feature_cols = feature_cols
        self.dm = device_manager
        self.seq_len = seq_len

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_name: str = "vol_transformer",
        base_dir: str = "v3",
        device: str = "cuda",
    ) -> "VolInference":
        """Load model from checkpoint for inference."""
        storage = StorageManager(base_dir=base_dir)
        dm = DeviceManager(compile_model=False)

        # Load feature config
        feat_cfg = storage.load_json("feature_config.json")
        feature_cols = feat_cfg["feature_cols"]

        # Load normalizer
        normalizer = FeatureNormalizer.load(f"{base_dir}/saved_models/normalizer_stats.json")

        # Load model
        checkpoint = storage.load_checkpoint(checkpoint_name)
        model = VolTransformer(
            input_dim=len(feature_cols),
            # Other params loaded from checkpoint or config
        )
        model.load_state_dict(checkpoint["state_dict"])

        return cls(model, normalizer, feature_cols, dm)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run per-ticker inference.

        Args:
            df: DataFrame with OHLCV + features. Must have 'ticker', 'date', and feature columns.

        Returns:
            DataFrame with columns: ticker, date, vol_score, confidence.
        """
        # Validate normalizer covers all feature columns
        if self.normalizer.is_fitted:
            missing = [c for c in self.feature_cols if c not in self.normalizer.mean_]
            if missing:
                logger.warning(f"Normalizer missing {len(missing)} features: {missing[:5]}. "
                               "Filling with 0 (z-score mean).")

        # Normalize features
        df_norm = self.normalizer.transform(df, self.feature_cols)

        results = []

        for ticker, group in df_norm.groupby("ticker"):
            group = group.sort_values("date")

            if len(group) < self.seq_len:
                continue

            # Take last seq_len rows
            features = group[self.feature_cols].values[-self.seq_len:]
            features = np.nan_to_num(features, nan=0.0).astype(np.float32)

            # Inference
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.dm.device)
            output = self.model(x)

            vol_score = output["prediction"].item()
            confidence = output.get("confidence")
            conf_val = confidence.item() if confidence is not None else 0.5

            results.append({
                "ticker": ticker,
                "date": group["date"].iloc[-1],
                "vol_score": vol_score,
                "confidence": conf_val,
            })

        result_df = pd.DataFrame(results)
        logger.info(f"Inference: {len(result_df)} tickers scored")

        return result_df

    @torch.no_grad()
    def predict_batch(self, sequences: torch.Tensor) -> dict[str, torch.Tensor]:
        """Batch inference on pre-prepared tensors.

        Args:
            sequences: (batch, seq_len, n_features)

        Returns:
            {"prediction": (batch,), "confidence": (batch,)}
        """
        sequences = sequences.to(self.dm.device)
        return self.model(sequences)
