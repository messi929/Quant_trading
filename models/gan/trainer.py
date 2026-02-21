"""WGAN-GP training with critic/generator alternation."""

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from models.gan.model import ConditionalWGAN
from utils.device import DeviceManager
from utils.storage import StorageManager


class GANTrainer:
    """Trains the Conditional WGAN-GP for market simulation."""

    def __init__(
        self,
        model: ConditionalWGAN,
        device_manager: DeviceManager,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        n_critic: int = 5,
        gp_weight: float = 10.0,
    ):
        self.model = model.to(device_manager.device)
        self.dm = device_manager
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.storage = StorageManager()

        self.optimizer_g = torch.optim.Adam(
            model.generator.parameters(), lr=lr_g, betas=(0.0, 0.9)
        )
        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9)
        )

    def _make_condition(
        self,
        sequences: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Create regime condition from sequence data.

        Uses simple heuristic: compare first-half and second-half
        returns + volatility to classify regime.
        """
        mid = sequences.size(1) // 2
        first_half = sequences[:, :mid, 0]  # First feature assumed to be return-like
        second_half = sequences[:, mid:, 0]

        conditions = torch.zeros(
            batch_size, ConditionalWGAN.N_REGIMES, device=sequences.device
        )

        for i in range(batch_size):
            ret = second_half[i].mean()
            vol = second_half[i].std()
            vol_med = first_half[i].std() * 1.5

            if vol > vol_med:
                regime = 3
            elif ret > 0.001:
                regime = 0
            elif ret < -0.001:
                regime = 1
            else:
                regime = 2
            conditions[i, regime] = 1.0

        return conditions

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        self.model.train()
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_gp = 0.0
        n_d_steps = 0
        n_g_steps = 0

        pbar = tqdm(train_loader, desc="GAN Train", leave=False)
        for step, (sequences, targets, sectors) in enumerate(pbar):
            sequences = sequences.to(self.dm.device)
            batch_size = sequences.size(0)
            condition = self._make_condition(sequences, batch_size)

            # === Train Discriminator ===
            noise = torch.randn(
                batch_size, self.model.noise_dim, device=self.dm.device
            )
            with torch.no_grad():
                fake = self.model.generator(noise, condition)

            d_real = self.model.discriminator(sequences, condition)
            d_fake = self.model.discriminator(fake, condition)

            gp = ConditionalWGAN.gradient_penalty(
                self.model.discriminator,
                sequences,
                fake,
                condition,
                self.dm.device,
            )

            d_loss = d_fake.mean() - d_real.mean() + self.gp_weight * gp

            self.optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            self.optimizer_d.step()

            total_d_loss += d_loss.item()
            total_gp += gp.item()
            n_d_steps += 1

            # === Train Generator every n_critic steps ===
            if (step + 1) % self.n_critic == 0:
                noise = torch.randn(
                    batch_size, self.model.noise_dim, device=self.dm.device
                )
                fake = self.model.generator(noise, condition)
                g_score = self.model.discriminator(fake, condition)
                g_loss = -g_score.mean()

                self.optimizer_g.zero_grad(set_to_none=True)
                g_loss.backward()
                self.optimizer_g.step()

                total_g_loss += g_loss.item()
                n_g_steps += 1

            pbar.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                gp=f"{gp.item():.4f}",
            )

        return {
            "d_loss": total_d_loss / max(1, n_d_steps),
            "g_loss": total_g_loss / max(1, n_g_steps),
            "gradient_penalty": total_gp / max(1, n_d_steps),
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate GAN quality with Wasserstein distance estimate."""
        self.model.eval()
        total_w_dist = 0.0
        n_batches = 0

        for sequences, targets, sectors in val_loader:
            sequences = sequences.to(self.dm.device)
            batch_size = sequences.size(0)
            condition = self._make_condition(sequences, batch_size)

            noise = torch.randn(
                batch_size, self.model.noise_dim, device=self.dm.device
            )
            fake = self.model.generator(noise, condition)

            d_real = self.model.discriminator(sequences, condition).mean()
            d_fake = self.model.discriminator(fake, condition).mean()

            total_w_dist += (d_real - d_fake).item()
            n_batches += 1

        return {"wasserstein_distance": total_w_dist / max(1, n_batches)}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        save_name: str = "gan",
        early_stop_patience: float = 5.0,
    ) -> dict:
        best_w_dist = float("inf")
        history = {"train": [], "val": []}

        logger.info(f"Starting WGAN-GP training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            w_dist = abs(val_metrics["wasserstein_distance"])

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"D Loss: {train_metrics['d_loss']:.4f} | "
                    f"G Loss: {train_metrics['g_loss']:.4f} | "
                    f"GP: {train_metrics['gradient_penalty']:.4f} | "
                    f"W-dist: {val_metrics['wasserstein_distance']:.4f}"
                )

            if w_dist < best_w_dist:
                best_w_dist = w_dist
                self.storage.save_model_checkpoint(
                    {
                        "generator": self.model.generator.state_dict(),
                        "discriminator": self.model.discriminator.state_dict(),
                    },
                    save_name,
                    epoch,
                    metrics=val_metrics,
                )

            # Early stopping: W-dist diverged too far from best
            if epoch > 5 and w_dist > best_w_dist + early_stop_patience:
                logger.info(
                    f"GAN early stopping at epoch {epoch}: "
                    f"W-dist {w_dist:.2f} > best {best_w_dist:.2f} + {early_stop_patience}"
                )
                break

            if epoch % 50 == 0:
                self.dm.log_memory(f"GAN Epoch {epoch}: ")

        return {"history": history, "best_wasserstein_distance": best_w_dist}

    def generate_synthetic_data(
        self,
        n_samples: int,
        regime: int,
    ) -> torch.Tensor:
        """Generate synthetic market data for RL augmentation."""
        self.model.eval()
        with torch.no_grad():
            data = self.model.generate(n_samples, regime, self.dm.device)
        return data.cpu()
