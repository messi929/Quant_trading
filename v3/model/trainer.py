"""VolTransformer trainer with deployment gates.

Loss: Huber (magnitude) + ListMLE (ranking) + BCE (confidence calibration)
Gates: vol_ic > 0.30, vol_rank_ic > 0.20, dir_acc > 0.55
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger
from tqdm import tqdm
from scipy.stats import spearmanr

from v3.model.vol_transformer import VolTransformer
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


class ListMLELoss(nn.Module):
    """Listwise ranking loss (chunked for efficiency)."""

    def __init__(self, temperature: float = 1.0, group_size: int = 64):
        super().__init__()
        self.temperature = temperature
        self.group_size = group_size

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = preds.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        # Shuffle + chunk
        idx = torch.randperm(n, device=preds.device)
        preds_s = preds[idx]
        targets_s = targets[idx]

        total_loss = torch.tensor(0.0, device=preds.device)
        n_groups = 0

        for i in range(0, n, self.group_size):
            p = preds_s[i : i + self.group_size] / self.temperature
            t = targets_s[i : i + self.group_size]
            if len(p) < 2:
                continue

            _, order = t.sort(descending=True)
            p_sorted = p[order]

            loss = torch.tensor(0.0, device=preds.device)
            for j in range(len(p_sorted) - 1):
                loss = loss - p_sorted[j] + torch.logsumexp(p_sorted[j:], dim=0)

            total_loss = total_loss + loss / len(p_sorted)
            n_groups += 1

        return total_loss / max(n_groups, 1)


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class VolTrainer:
    """Trains VolTransformer with vol-specific losses and deployment gates."""

    def __init__(
        self,
        model: VolTransformer,
        device_manager: DeviceManager,
        learning_rate: float = 3e-5,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 1e-5,
        ranking_loss_weight: float = 0.6,
        confidence_loss_weight: float = 0.1,
        min_vol_ic: float = 0.30,
        min_vol_rank_ic: float = 0.20,
        min_dir_acc: float = 0.55,
    ):
        self.model = device_manager.prepare_model(model)
        self.dm = device_manager
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.storage = StorageManager(base_dir="v3")

        # Loss functions
        self.criterion_huber = nn.HuberLoss(delta=1.5)
        self.criterion_rank = ListMLELoss(temperature=1.0, group_size=64)
        self.rank_weight = ranking_loss_weight
        self.conf_weight = confidence_loss_weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

        # Gates
        self.min_vol_ic = min_vol_ic
        self.min_vol_rank_ic = min_vol_rank_ic
        self.min_dir_acc = min_dir_acc

    def train_epoch(self, train_loader: DataLoader, scheduler: LambdaLR) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for step, batch in enumerate(pbar):
            features = batch["features"].to(self.dm.device)
            targets = batch["target"].to(self.dm.device)

            with self.dm.autocast():
                output = self.model(features)
                preds = output["prediction"]
                huber = self.criterion_huber(preds, targets)
                rank = self.criterion_rank(preds, targets)
                loss = (1 - self.rank_weight) * huber + self.rank_weight * rank

            # Confidence loss outside autocast
            confidence = output.get("confidence")
            if confidence is not None and self.conf_weight > 0:
                direction_correct = ((preds.detach() > 0) == (targets > 0)).float()
                conf_loss = F.binary_cross_entropy(confidence.float(), direction_correct)
                loss = loss + self.conf_weight * conf_loss

            loss = loss / self.grad_accum
            is_accumulating = (step + 1) % self.grad_accum != 0
            self.dm.backward_step(
                loss, self.optimizer, self.max_grad_norm, accumulation_step=is_accumulating,
            )

            if not is_accumulating:
                scheduler.step()

            total_loss += loss.item() * self.grad_accum
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets, all_confs = [], [], []
        n_batches = 0

        for batch in loader:
            features = batch["features"].to(self.dm.device)
            targets = batch["target"].to(self.dm.device)

            with self.dm.autocast():
                output = self.model(features)
                preds = output["prediction"]
                huber = self.criterion_huber(preds, targets)
                rank = self.criterion_rank(preds, targets)
                loss = (1 - self.rank_weight) * huber + self.rank_weight * rank

            confidence = output.get("confidence")
            if confidence is not None and self.conf_weight > 0:
                direction_correct = ((preds > 0) == (targets > 0)).float()
                conf_loss = F.binary_cross_entropy(confidence.float(), direction_correct)
                loss = loss + self.conf_weight * conf_loss

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            if confidence is not None:
                all_confs.append(confidence.cpu())
            n_batches += 1

        preds = torch.cat(all_preds)
        targs = torch.cat(all_targets)

        # Direction accuracy (vol expansion direction)
        dir_acc = ((preds > 0) == (targs > 0)).float().mean().item()

        # Rank IC (Spearman)
        ic_val, _ = spearmanr(preds.numpy(), targs.numpy())
        if ic_val != ic_val:
            ic_val = 0.0

        # Pearson IC
        pearson_ic = float(torch.corrcoef(torch.stack([preds, targs]))[0, 1])
        if pearson_ic != pearson_ic:
            pearson_ic = 0.0

        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "dir_acc": dir_acc,
            "vol_rank_ic": float(ic_val),
            "vol_ic": pearson_ic,
        }

        # Confidence calibration
        if all_confs:
            confs = torch.cat(all_confs)
            direction_correct = ((preds > 0) == (targs > 0)).float()
            high_mask = confs > 0.6
            if high_mask.sum() > 10:
                metrics["high_conf_acc"] = direction_correct[high_mask].mean().item()
                metrics["low_conf_acc"] = direction_correct[~high_mask].mean().item()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 60,
        patience: int = 20,
        save_name: str = "vol_transformer",
    ) -> dict:
        total_steps = epochs * len(train_loader) // self.grad_accum
        scheduler = cosine_warmup_scheduler(self.optimizer, self.warmup_steps, total_steps)

        best_val_loss = float("inf")
        best_metrics = {}
        patience_counter = 0
        history = {"train": [], "val": []}

        logger.info(f"VolTransformer training: {epochs} epochs, "
                     f"{self.model.count_parameters():,} parameters")
        logger.info(f"Gates: vol_ic>{self.min_vol_ic}, rank_ic>{self.min_vol_rank_ic}, "
                     f"dir_acc>{self.min_dir_acc}")

        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(train_loader, scheduler)
            val_m = self.validate(val_loader)
            history["train"].append(train_m)
            history["val"].append(val_m)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {train_m['loss']:.4f}/{val_m['val_loss']:.4f} | "
                f"Dir: {val_m['dir_acc']:.4f} | "
                f"Vol IC: {val_m['vol_ic']:.4f} | "
                f"Rank IC: {val_m['vol_rank_ic']:.4f}"
            )

            if val_m["val_loss"] < best_val_loss:
                best_val_loss = val_m["val_loss"]
                best_metrics = val_m.copy()
                patience_counter = 0
                self.storage.save_checkpoint(
                    self.model.state_dict(), save_name, epoch, metrics=val_m,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        gate_pass = self._check_gates(best_metrics)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "gate_pass": gate_pass,
        }

    def _check_gates(self, metrics: dict) -> bool:
        vol_ic = metrics.get("vol_ic", 0)
        rank_ic = metrics.get("vol_rank_ic", 0)
        dir_acc = metrics.get("dir_acc", 0)

        pass_ic = vol_ic >= self.min_vol_ic
        pass_rank = rank_ic >= self.min_vol_rank_ic
        pass_dir = dir_acc >= self.min_dir_acc

        logger.info("=" * 60)
        logger.info("V3 DEPLOYMENT GATE CHECK")
        logger.info("=" * 60)
        logger.info(f"  Vol IC:      {vol_ic:.4f}  {'PASS' if pass_ic else 'FAIL'} (>{self.min_vol_ic})")
        logger.info(f"  Vol Rank IC: {rank_ic:.4f}  {'PASS' if pass_rank else 'FAIL'} (>{self.min_vol_rank_ic})")
        logger.info(f"  Dir Acc:     {dir_acc:.4f}  {'PASS' if pass_dir else 'FAIL'} (>{self.min_dir_acc})")

        all_pass = pass_ic and pass_rank and pass_dir
        if all_pass:
            logger.info("  RESULT: ALL GATES PASSED")
        else:
            logger.warning("  RESULT: GATE FAILED — do NOT proceed to backtest")
        logger.info("=" * 60)

        return all_pass

    def evaluate_test(self, test_loader: DataLoader) -> dict:
        metrics = self.validate(test_loader)
        logger.info("=" * 60)
        logger.info("TEST SET EVALUATION (out-of-sample)")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        gate_pass = self._check_gates(metrics)
        metrics["gate_pass"] = gate_pass
        logger.info("=" * 60)
        return metrics
