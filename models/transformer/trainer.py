"""Transformer training with gradient checkpointing and cosine warmup."""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger
from tqdm import tqdm

from models.transformer.model import TemporalTransformer
from utils.device import DeviceManager
from utils.storage import StorageManager


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class TransformerTrainer:
    """Trains the Temporal Transformer for return prediction."""

    def __init__(
        self,
        model: TemporalTransformer,
        device_manager: DeviceManager,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        weight_decay: float = 1e-5,
    ):
        self.model = device_manager.prepare_model(model)
        self.dm = device_manager
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.storage = StorageManager()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.5)  # Robust to outlier returns

    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: LambdaLR,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for step, (sequences, targets, sectors) in enumerate(pbar):
            sequences = sequences.to(self.dm.device)
            targets = targets.to(self.dm.device)
            sectors = sectors.to(self.dm.device)

            with self.dm.autocast():
                output = self.model(sequences, sectors)
                loss = self.criterion(output["prediction"], targets)
                loss = loss / self.grad_accum

            is_accumulating = (step + 1) % self.grad_accum != 0
            self.dm.backward_step(
                loss,
                self.optimizer,
                max_grad_norm=self.max_grad_norm,
                accumulation_step=is_accumulating,
            )

            if not is_accumulating:
                scheduler.step()

            total_loss += loss.item() * self.grad_accum
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        for sequences, targets, sectors in val_loader:
            sequences = sequences.to(self.dm.device)
            targets = targets.to(self.dm.device)
            sectors = sectors.to(self.dm.device)

            with self.dm.autocast():
                output = self.model(sequences, sectors)
                loss = self.criterion(output["prediction"], targets)

            total_loss += loss.item()
            all_preds.append(output["prediction"].cpu())
            all_targets.append(targets.cpu())
            n_batches += 1

        # Compute directional accuracy
        preds = torch.cat(all_preds)
        targs = torch.cat(all_targets)
        direction_acc = ((preds > 0) == (targs > 0)).float().mean().item()

        # Rank IC (Spearman correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(preds.numpy(), targs.numpy())

        return {
            "val_loss": total_loss / n_batches,
            "direction_accuracy": direction_acc,
            "rank_ic": ic if not (ic != ic) else 0.0,  # NaN check
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 80,
        patience: int = 15,
        save_name: str = "transformer",
    ) -> dict:
        total_steps = epochs * len(train_loader) // self.grad_accum
        scheduler = cosine_warmup_scheduler(
            self.optimizer, self.warmup_steps, total_steps
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train": [], "val": []}

        logger.info(f"Starting Transformer training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, scheduler)
            val_metrics = self.validate(val_loader)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Dir Acc: {val_metrics['direction_accuracy']:.4f} | "
                f"IC: {val_metrics['rank_ic']:.4f}"
            )

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
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.dm.log_memory(f"Epoch {epoch}: ")

        return {"history": history, "best_val_loss": best_val_loss}
