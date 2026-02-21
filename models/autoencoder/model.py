"""Variational Autoencoder for latent indicator discovery.

The VAE compresses market features into a latent space where each
dimension represents a "discovered indicator" — a learned representation
that captures non-linear relationships in market data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes market features into latent space parameters (mu, log_var)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Decodes latent vectors back to market feature space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(hidden_dims[0], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class MarketVAE(nn.Module):
    """Variational Autoencoder for market feature compression.

    Latent dimensions serve as "discovered indicators" that capture
    non-linear market dynamics not present in traditional indicators.

    Architecture:
        Input (n_features) → Encoder → (mu, log_var) → z → Decoder → Output (n_features)

    The reparameterization trick allows gradient flow through the
    stochastic sampling step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        latent_dim: int = 32,
        dropout: float = 0.2,
        seq_length: int = 60,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.input_dim = input_dim

        # Flatten seq_length * input_dim for the encoder
        flat_dim = seq_length * input_dim

        self.encoder = Encoder(flat_dim, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, flat_dim, dropout)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # Use mean during inference

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_length, n_features) market sequence

        Returns:
            Dict with keys: reconstruction, mu, log_var, z
        """
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)

        mu, log_var = self.encoder(x_flat)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        x_recon = x_recon.reshape(batch_size, self.seq_length, self.input_dim)

        return {
            "reconstruction": x_recon,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (returns mu for deterministic encoding).

        Args:
            x: (batch, seq_length, n_features)

        Returns:
            z: (batch, latent_dim) — the "discovered indicators"
        """
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        mu, _ = self.encoder(x_flat)
        return mu

    @staticmethod
    def loss_function(
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 0.001,
    ) -> dict[str, torch.Tensor]:
        """VAE loss = Reconstruction + KL divergence.

        Args:
            x: Original input
            recon_x: Reconstructed output
            mu: Latent mean
            log_var: Latent log variance
            kl_weight: Beta parameter for beta-VAE

        Returns:
            Dict with total_loss, recon_loss, kl_loss
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
