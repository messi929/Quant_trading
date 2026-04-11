"""GPU management, mixed precision, and memory optimization."""

import gc
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class DeviceManager:
    """Manages GPU resources and mixed precision training."""

    def __init__(self, memory_fraction: float = 0.85, compile_model: bool = False):
        self.compile_model = compile_model
        self.device = self._setup_device(memory_fraction)
        self.scaler = torch.amp.GradScaler("cuda") if self.is_cuda else None

    def _setup_device(self, memory_fraction: float) -> torch.device:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({total_mem:.1f} GB)")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        return device

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.compile_model and self.is_cuda:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed, using eager mode: {e}")
        return model

    @contextmanager
    def autocast(self):
        if self.is_cuda:
            with torch.amp.autocast("cuda"):
                yield
        else:
            yield

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: Optional[float] = 1.0,
        accumulation_step: bool = False,
    ) -> float:
        loss_val = loss.item()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if not accumulation_step:
                self.scaler.unscale_(optimizer)
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self._get_params(optimizer), max_grad_norm
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if not accumulation_step:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self._get_params(optimizer), max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return loss_val

    @staticmethod
    def _get_params(optimizer: torch.optim.Optimizer):
        params = []
        for group in optimizer.param_groups:
            params.extend(group["params"])
        return params

    def clear_memory(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
            gc.collect()

    def memory_stats(self) -> dict:
        if not self.is_cuda:
            return {"device": "cpu"}
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
