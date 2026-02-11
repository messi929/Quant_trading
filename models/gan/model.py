"""Conditional WGAN-GP for market data simulation.

Generates synthetic market sequences conditioned on market regime
(bull/bear/sideways/volatile) for RL training data augmentation.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generates synthetic market sequences from noise + regime condition."""

    def __init__(
        self,
        noise_dim: int = 64,
        condition_dim: int = 4,
        hidden_dims: list[int] = None,
        output_dim: int = 32,
        seq_length: int = 60,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256, 128]
        self.seq_length = seq_length
        self.output_dim = output_dim

        input_dim = noise_dim + condition_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(hidden_dims[-1], seq_length * output_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noise: (batch, noise_dim) random noise
            condition: (batch, condition_dim) one-hot market regime

        Returns:
            (batch, seq_length, output_dim) synthetic market sequence
        """
        x = torch.cat([noise, condition], dim=-1)
        out = self.network(x)
        return out.view(-1, self.seq_length, self.output_dim)


class Discriminator(nn.Module):
    """Discriminates real vs. synthetic market sequences."""

    def __init__(
        self,
        input_dim: int = 32,
        condition_dim: int = 4,
        hidden_dims: list[int] = None,
        seq_length: int = 60,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256, 128]

        flat_dim = seq_length * input_dim + condition_dim
        layers = []
        in_dim = flat_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim

        # WGAN: no sigmoid, output is a scalar "realness" score
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)
        self.seq_length = seq_length

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, input_dim) market sequence
            condition: (batch, condition_dim) one-hot market regime

        Returns:
            (batch, 1) realness score
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_cond = torch.cat([x_flat, condition], dim=-1)
        return self.network(x_cond)


class ConditionalWGAN(nn.Module):
    """Conditional Wasserstein GAN with Gradient Penalty.

    Market regime conditions:
        0: Bull (상승장)
        1: Bear (하락장)
        2: Sideways (횡보장)
        3: Volatile (고변동장)
    """

    N_REGIMES = 4

    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dims: list[int] = None,
        output_dim: int = 32,
        seq_length: int = 60,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256, 128]
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

        self.generator = Generator(
            noise_dim, self.N_REGIMES, hidden_dims, output_dim, seq_length
        )
        self.discriminator = Discriminator(
            output_dim, self.N_REGIMES, hidden_dims, seq_length
        )

    def generate(
        self,
        batch_size: int,
        regime: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate synthetic market data for a specific regime.

        Args:
            batch_size: Number of sequences to generate
            regime: Market regime index (0-3)
            device: Target device

        Returns:
            (batch_size, seq_length, output_dim) synthetic data
        """
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        condition = torch.zeros(batch_size, self.N_REGIMES, device=device)
        condition[:, regime] = 1.0
        return self.generator(noise, condition)

    @staticmethod
    def gradient_penalty(
        discriminator: Discriminator,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP training stability."""
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interpolated = discriminator(interpolated, condition)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    @staticmethod
    def classify_regime(returns: torch.Tensor, volatility: torch.Tensor) -> int:
        """Classify current market regime from returns and volatility.

        Args:
            returns: Recent period returns
            volatility: Recent period volatility

        Returns:
            Regime index (0=bull, 1=bear, 2=sideways, 3=volatile)
        """
        avg_return = returns.mean().item()
        avg_vol = volatility.mean().item()
        vol_threshold = volatility.median().item() * 1.5

        if avg_vol > vol_threshold:
            return 3  # Volatile
        elif avg_return > 0.001:
            return 0  # Bull
        elif avg_return < -0.001:
            return 1  # Bear
        else:
            return 2  # Sideways
