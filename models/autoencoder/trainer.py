"""VAE training with KL annealing and gradient accumulation."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from loguru import logger
from tqdm import tqdm

from models.autoencoder.model import MarketVAE
from utils.device import DeviceManager
from utils.storage import StorageManager


class VAETrainer:
    """Trains the MarketVAE with KL annealing and mixed precision."""

    def __init__(
        self,
        model: MarketVAE,
        device_manager: DeviceManager,
        learning_rate: float = 1e-4,
        kl_weight: float = 0.001,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 20,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        weight_decay: float = 1e-5,
    ):
        self.model = device_manager.prepare_model(model)
        self.dm = device_manager
        self.kl_weight_target = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.storage = StorageManager()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _get_kl_weight(self, epoch: int) -> float:
        """Linear KL annealing from 0 to target weight."""
        if not self.kl_annealing:
            return self.kl_weight_target
        return min(
            self.kl_weight_target,
            self.kl_weight_target * epoch / max(1, self.kl_annealing_epochs),
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train one epoch."""
        self.model.train()
        kl_weight = self._get_kl_weight(epoch)

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for step, (sequences, targets, sectors) in enumerate(pbar):
            sequences = sequences.to(self.dm.device)

            with self.dm.autocast():
                output = self.model(sequences)
                losses = MarketVAE.loss_function(
                    sequences,
                    output["reconstruction"],
                    output["mu"],
                    output["log_var"],
                    kl_weight=kl_weight,
                )
                loss = losses["total_loss"] / self.grad_accum

            is_accumulating = (step + 1) % self.grad_accum != 0
            self.dm.backward_step(
                loss,
                self.optimizer,
                max_grad_norm=self.max_grad_norm,
                accumulation_step=is_accumulating,
            )

            total_loss += losses["total_loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                recon=f"{losses['recon_loss'].item():.4f}",
                kl=f"{losses['kl_loss'].item():.4f}",
            )

        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "kl_loss": total_kl / n_batches,
            "kl_weight": kl_weight,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Validate model."""
        self.model.eval()
        kl_weight = self._get_kl_weight(epoch)

        total_loss = 0.0
        total_recon = 0.0
        n_batches = 0

        for sequences, targets, sectors in val_loader:
            sequences = sequences.to(self.dm.device)

            with self.dm.autocast():
                output = self.model(sequences)
                losses = MarketVAE.loss_function(
                    sequences,
                    output["reconstruction"],
                    output["mu"],
                    output["log_var"],
                    kl_weight=kl_weight,
                )

            total_loss += losses["total_loss"].item()
            total_recon += losses["recon_loss"].item()
            n_batches += 1

        return {
            "val_loss": total_loss / n_batches,
            "val_recon_loss": total_recon / n_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        save_name: str = "vae",
    ) -> dict:
        """Full training loop with early stopping.

        Returns:
            Dict with training history and best metrics.
        """
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train": [], "val": []}

        logger.info(f"Starting VAE training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            scheduler.step()

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(recon: {train_metrics['recon_loss']:.4f}, "
                f"kl: {train_metrics['kl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"KL weight: {train_metrics['kl_weight']:.5f}"
            )

            # Early stopping
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.storage.save_model_checkpoint(
                    self.model.state_dict(),
                    save_name,
                    epoch,
                    metrics=val_metrics,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best val_loss: {best_val_loss:.4f})"
                    )
                    break

            self.dm.log_memory(f"Epoch {epoch}: ")

        return {
            "history": history,
            "best_val_loss": best_val_loss,
        }

    @torch.no_grad()
    def extract_latent_features(
        self,
        dataloader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract latent features (discovered indicators) from data.

        Returns:
            Tuple of (latent_features, sector_ids)
        """
        self.model.eval()
        all_z = []
        all_sectors = []

        for sequences, targets, sectors in dataloader:
            sequences = sequences.to(self.dm.device)
            z = self.model.encode(sequences)
            all_z.append(z.cpu())
            all_sectors.append(sectors)

        return torch.cat(all_z, dim=0), torch.cat(all_sectors, dim=0)
