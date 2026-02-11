"""Hybrid model ensemble orchestrating VAE, Transformer, GAN, and RL."""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from models.autoencoder.model import MarketVAE
from models.transformer.model import TemporalTransformer
from models.gan.model import ConditionalWGAN
from models.rl.agent import PPOAgent


class AttentionFusion(nn.Module):
    """Attention-based fusion of multi-model signals."""

    def __init__(self, signal_dims: list[int], output_dim: int):
        super().__init__()
        total_dim = sum(signal_dims)
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.Tanh(),
            nn.Linear(total_dim // 2, len(signal_dims)),
            nn.Softmax(dim=-1),
        )
        # Project each signal to common dim
        self.projections = nn.ModuleList([
            nn.Linear(d, output_dim) for d in signal_dims
        ])

    def forward(self, signals: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            signals: List of (batch, signal_dim_i) tensors

        Returns:
            (batch, output_dim) fused signal
        """
        concat = torch.cat(signals, dim=-1)
        weights = self.attention(concat)  # (batch, n_signals)

        projected = [proj(s) for proj, s in zip(self.projections, signals)]
        stacked = torch.stack(projected, dim=1)  # (batch, n_signals, output_dim)
        weights = weights.unsqueeze(-1)  # (batch, n_signals, 1)

        fused = (stacked * weights).sum(dim=1)  # (batch, output_dim)
        return fused


class ModelEnsemble:
    """Orchestrates the full model pipeline for inference.

    Pipeline:
        1. VAE: raw features → latent indicators
        2. Transformer: latent sequences → signal embeddings + predictions
        3. GAN: generate augmented scenarios (training only)
        4. Fusion: combine VAE latent + Transformer signals
        5. RL Agent: fused signals → sector allocations
    """

    def __init__(
        self,
        vae: MarketVAE,
        transformer: TemporalTransformer,
        gan: ConditionalWGAN,
        rl_agent: PPOAgent,
        fusion_output_dim: int = 128,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.vae = vae.to(self.device)
        self.transformer = transformer.to(self.device)
        self.gan = gan.to(self.device)
        self.rl_agent = rl_agent

        # Attention fusion layer
        vae_dim = vae.latent_dim
        transformer_dim = transformer.d_model
        self.fusion = AttentionFusion(
            signal_dims=[vae_dim, transformer_dim],
            output_dim=fusion_output_dim,
        ).to(self.device)

    @torch.no_grad()
    def get_signals(
        self,
        sequences: torch.Tensor,
        sector_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract signals from all models.

        Args:
            sequences: (batch, seq_len, n_features)
            sector_ids: (batch,) sector IDs

        Returns:
            Dict of signal tensors
        """
        self.vae.eval()
        self.transformer.eval()

        sequences = sequences.to(self.device)
        sector_ids = sector_ids.to(self.device)

        # VAE: extract latent indicators
        latent = self.vae.encode(sequences)  # (batch, latent_dim)

        # Transformer: get signal embedding
        tf_output = self.transformer(sequences, sector_ids)
        signal_emb = tf_output["signal_embedding"]  # (batch, d_model)
        prediction = tf_output["prediction"]  # (batch,)

        # Fusion
        fused = self.fusion([latent, signal_emb])

        return {
            "latent_indicators": latent,
            "signal_embedding": signal_emb,
            "prediction": prediction,
            "fused_signal": fused,
        }

    def get_allocation(
        self,
        sequences: torch.Tensor,
        sector_ids: torch.Tensor,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get sector allocation from the full pipeline.

        Args:
            sequences: (1, seq_len, n_features) single observation
            sector_ids: (1,) sector ID

        Returns:
            (n_sectors,) allocation weights
        """
        signals = self.get_signals(sequences, sector_ids)

        # Build RL state from fused signals
        fused = signals["fused_signal"].squeeze(0).cpu().numpy()

        # RL agent decides allocation
        action, _, _ = self.rl_agent.select_action(fused, deterministic)
        return action

    def generate_augmented_data(
        self,
        regime: int,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Generate synthetic data using GAN for specific market regime."""
        self.gan.eval()
        with torch.no_grad():
            return self.gan.generate(n_samples, regime, self.device)

    def save_ensemble(self, path: str) -> None:
        """Save all model states."""
        import torch as _torch

        _torch.save(
            {
                "vae": self.vae.state_dict(),
                "transformer": self.transformer.state_dict(),
                "gan_generator": self.gan.generator.state_dict(),
                "gan_discriminator": self.gan.discriminator.state_dict(),
                "rl_policy": self.rl_agent.policy.state_dict(),
                "fusion": self.fusion.state_dict(),
            },
            path,
        )
        logger.info(f"Ensemble saved to {path}")

    def load_ensemble(self, path: str) -> None:
        """Load all model states."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(checkpoint["vae"])
        self.transformer.load_state_dict(checkpoint["transformer"])
        self.gan.generator.load_state_dict(checkpoint["gan_generator"])
        self.gan.discriminator.load_state_dict(checkpoint["gan_discriminator"])
        self.rl_agent.policy.load_state_dict(checkpoint["rl_policy"])
        self.fusion.load_state_dict(checkpoint["fusion"])
        logger.info(f"Ensemble loaded from {path}")
