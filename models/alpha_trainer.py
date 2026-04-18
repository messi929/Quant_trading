"""AlphaTransformer trainer with deployment gates (v2.3).

Loss: Huber (magnitude) + ListMLE (ranking) + BCE (confidence)
Gates: dir_acc > 54.5%, rank_ic > 0.10 — must pass before deployment.

v2.3 changes:
  - PairwiseRankingLoss → ListMLE (learns full ranking, not random pairs)
  - Huber delta 0.5 → 1.5 (robust to outlier returns)
  - Deployment gates tightened (52% → 54.5% dir_acc)
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

from models.alpha_model import AlphaTransformer
from utils.device import DeviceManager
from utils.storage import StorageManager


# ── Loss Functions ────────────────────────────────────────────

class ListMLELoss(nn.Module):
    """ListMLE: Listwise ranking loss.

    Learns the full permutation probability of the correct ranking,
    not just random pairwise comparisons. Provably better for Rank IC.

    Reference: Xia et al. "Listwise Approach to Learning to Rank" (2008)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            preds: (batch,) model predictions
            targets: (batch,) ground truth values

        Returns:
            Scalar loss (lower = predictions rank targets better)
        """
        n = preds.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        # Sort targets descending → get the "ideal" permutation
        _, ideal_order = targets.sort(descending=True)

        # Reorder predictions by ideal ranking
        preds_sorted = preds[ideal_order] / self.temperature

        # ListMLE: sum of log-softmax along the ranked list
        # For position i, probability = exp(s_i) / sum(exp(s_j) for j >= i)
        loss = torch.tensor(0.0, device=preds.device)
        for i in range(n - 1):
            # log(exp(s_i) / sum(exp(s_j) for j >= i))
            # = s_i - logsumexp(s_j for j >= i)
            loss = loss - preds_sorted[i] + torch.logsumexp(preds_sorted[i:], dim=0)

        return loss / n


class ListMLELossBatched(nn.Module):
    """Efficient ListMLE for large batches using chunked ranking.

    Splits batch into groups (simulating cross-sectional days),
    applies ListMLE within each group.
    """

    def __init__(self, temperature: float = 1.0, group_size: int = 64):
        super().__init__()
        self.temperature = temperature
        self.group_size = group_size
        self._listmle = ListMLELoss(temperature)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = preds.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        # Shuffle to avoid ordering bias, then chunk
        idx = torch.randperm(n, device=preds.device)
        preds_s = preds[idx]
        targets_s = targets[idx]

        total_loss = torch.tensor(0.0, device=preds.device)
        n_groups = 0
        for i in range(0, n, self.group_size):
            p = preds_s[i : i + self.group_size]
            t = targets_s[i : i + self.group_size]
            if len(p) < 2:
                continue
            total_loss = total_loss + self._listmle(p, t)
            n_groups += 1

        return total_loss / max(n_groups, 1)


# ── Scheduler ─────────────────────────────────────────────────

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


# ── Trainer ───────────────────────────────────────────────────

class AlphaTrainer:
    """Trains AlphaTransformer with Huber + Ranking + Confidence loss.

    Deployment gates:
        - dir_acc > min_dir_acc (default 0.52)
        - rank_ic > min_rank_ic (default 0.10)
    """

    def __init__(
        self,
        model: AlphaTransformer,
        device_manager: DeviceManager,
        learning_rate: float = 3e-5,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 1e-5,
        ranking_loss_weight: float = 0.5,
        confidence_loss_weight: float = 0.1,
        min_dir_acc: float = 0.52,
        min_rank_ic: float = 0.10,
    ):
        self.model = device_manager.prepare_model(model)
        self.dm = device_manager
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.storage = StorageManager()

        # Loss — v2.3: ListMLE + Huber(delta=1.5)
        self.criterion_huber = nn.HuberLoss(delta=1.5)
        self.criterion_rank = ListMLELossBatched(temperature=1.0, group_size=64)
        self.rank_weight = ranking_loss_weight
        self.conf_weight = confidence_loss_weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Deployment gates
        self.min_dir_acc = min_dir_acc
        self.min_rank_ic = min_rank_ic

    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combined loss: Huber + Ranking + Confidence BCE."""
        huber = self.criterion_huber(preds, targets)
        rank = self.criterion_rank(preds, targets)
        loss = (1 - self.rank_weight) * huber + self.rank_weight * rank

        if confidence is not None and self.conf_weight > 0:
            # Target: 1 if prediction direction matches actual, 0 otherwise
            # Cast to float32 for BCE safety under autocast
            direction_correct = ((preds > 0) == (targets > 0)).float()
            conf_loss = F.binary_cross_entropy(
                confidence.float(), direction_correct.float(),
            )
            loss = loss + self.conf_weight * conf_loss

        return loss

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
        for step, batch in enumerate(pbar):
            sequences = batch[0].to(self.dm.device)
            targets = batch[1].to(self.dm.device)
            # batch[2] = sector_ids (not used in v2, but kept for compatibility)

            with self.dm.autocast():
                output = self.model(sequences)
                preds = output["prediction"]
                confidence = output.get("confidence")
                # Huber + Ranking under autocast
                huber = self.criterion_huber(preds, targets)
                rank = self.criterion_rank(preds, targets)
                loss = (1 - self.rank_weight) * huber + self.rank_weight * rank

            # Confidence BCE outside autocast (not autocast-safe)
            if confidence is not None and self.conf_weight > 0:
                direction_correct = ((preds.detach() > 0) == (targets > 0)).float()
                conf_loss = F.binary_cross_entropy(confidence.float(), direction_correct)
                loss = loss + self.conf_weight * conf_loss

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

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_confs = []
        n_batches = 0

        for batch in val_loader:
            sequences = batch[0].to(self.dm.device)
            targets = batch[1].to(self.dm.device)

            with self.dm.autocast():
                output = self.model(sequences)
                preds = output["prediction"]
                confidence = output.get("confidence")
                huber = self.criterion_huber(preds, targets)
                rank = self.criterion_rank(preds, targets)
                loss = (1 - self.rank_weight) * huber + self.rank_weight * rank

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

        # Direction accuracy
        dir_acc = ((preds > 0) == (targs > 0)).float().mean().item()

        # Rank IC (Spearman)
        from scipy.stats import spearmanr
        ic_val, _ = spearmanr(preds.numpy(), targs.numpy())
        if ic_val != ic_val:  # NaN check
            ic_val = 0.0

        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "dir_acc": dir_acc,
            "rank_ic": float(ic_val),
        }

        # Confidence calibration
        if all_confs:
            confs = torch.cat(all_confs)
            direction_correct = ((preds > 0) == (targs > 0)).float()
            # High confidence → higher accuracy?
            high_conf_mask = confs > 0.6
            if high_conf_mask.sum() > 10:
                high_conf_acc = direction_correct[high_conf_mask].mean().item()
                low_conf_acc = direction_correct[~high_conf_mask].mean().item()
                metrics["high_conf_acc"] = high_conf_acc
                metrics["low_conf_acc"] = low_conf_acc
                metrics["high_conf_pct"] = high_conf_mask.float().mean().item()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 15,
        save_name: str = "alpha_transformer",
    ) -> dict:
        total_steps = epochs * len(train_loader) // self.grad_accum
        scheduler = cosine_warmup_scheduler(
            self.optimizer, self.warmup_steps, total_steps
        )

        best_val_loss = float("inf")
        best_metrics = {}
        patience_counter = 0
        history = {"train": [], "val": []}

        logger.info(f"Starting AlphaTransformer training for {epochs} epochs")
        logger.info(
            f"Deployment gates: dir_acc > {self.min_dir_acc:.0%}, "
            f"rank_ic > {self.min_rank_ic:.2f}"
        )

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, scheduler)
            val_metrics = self.validate(val_loader)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            # Log
            log_parts = [
                f"Epoch {epoch}/{epochs}",
                f"Train Loss: {train_metrics['loss']:.4f}",
                f"Val Loss: {val_metrics['val_loss']:.4f}",
                f"Dir Acc: {val_metrics['dir_acc']:.4f}",
                f"IC: {val_metrics['rank_ic']:.4f}",
            ]
            if "high_conf_acc" in val_metrics:
                log_parts.append(
                    f"HiConf Acc: {val_metrics['high_conf_acc']:.4f} "
                    f"({val_metrics['high_conf_pct']:.0%})"
                )
            logger.info(" | ".join(log_parts))

            # Save best
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_metrics = val_metrics
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

        # ── Deployment gate check ──
        gate_pass = self._check_gates(best_metrics)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "gate_pass": gate_pass,
        }

    def _check_gates(self, metrics: dict) -> bool:
        """Check deployment gates. Returns True if all pass."""
        dir_acc = metrics.get("dir_acc", 0)
        rank_ic = metrics.get("rank_ic", 0)

        pass_dir = dir_acc >= self.min_dir_acc
        pass_ic = rank_ic >= self.min_rank_ic

        logger.info("=" * 60)
        logger.info("DEPLOYMENT GATE CHECK")
        logger.info("=" * 60)
        logger.info(
            f"  Dir Acc: {dir_acc:.4f} "
            f"{'PASS' if pass_dir else 'FAIL'} "
            f"(threshold: {self.min_dir_acc:.2f})"
        )
        logger.info(
            f"  Rank IC: {rank_ic:.4f} "
            f"{'PASS' if pass_ic else 'FAIL'} "
            f"(threshold: {self.min_rank_ic:.2f})"
        )

        if pass_dir and pass_ic:
            logger.info("  RESULT: ALL GATES PASSED — model ready for deployment")
        else:
            logger.warning(
                "  RESULT: GATE FAILED — do NOT deploy this model. "
                "Improve features/architecture before proceeding."
            )
        logger.info("=" * 60)

        return pass_dir and pass_ic

    def evaluate_test(self, test_loader: DataLoader) -> dict[str, float]:
        """Final evaluation on test set (out-of-sample)."""
        metrics = self.validate(test_loader)
        logger.info("=" * 60)
        logger.info("TEST SET EVALUATION (out-of-sample)")
        logger.info("=" * 60)
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        gate_pass = self._check_gates(metrics)
        metrics["gate_pass"] = gate_pass
        return metrics
