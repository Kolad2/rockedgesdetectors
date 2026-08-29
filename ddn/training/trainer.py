"""Reusable DDN training and validation loop."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .loss import DDNLoss


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    bce: float
    kl: float
    positive_pixels: float
    batches: int


class DDNTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: DDNLoss,
        device: torch.device,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
    ):
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.accumulation_steps = int(accumulation_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = torch.amp.GradScaler(
            device.type,
            enabled=self.use_amp,
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = _MetricTotals()
        accumulated = 0
        progress = tqdm(loader, desc=f"DDN train {epoch}", dynamic_ncols=True)

        for batch_index, batch in enumerate(progress, start=1):
            loss, components = self._forward_loss(batch, sample=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "DDN produced a non-finite loss. Disable AMP first; "
                    "float16 attention can overflow on older GPUs."
                )
            self.scaler.scale(loss).backward()
            accumulated += 1
            totals.update(loss, components)

            if accumulated == self.accumulation_steps or batch_index == len(loader):
                self._optimizer_step(accumulated)
                accumulated = 0

            progress.set_postfix(
                loss=f"{totals.loss / totals.batches:.2f}",
                bce=f"{totals.bce / totals.batches:.2f}",
                kl=f"{totals.kl / totals.batches:.2f}",
            )
        return totals.finish()

    @torch.inference_mode()
    def validate(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        self.model.eval()
        totals = _MetricTotals()
        progress = tqdm(loader, desc=f"DDN val {epoch}", dynamic_ncols=True)
        for batch in progress:
            loss, components = self._forward_loss(batch, sample=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("DDN produced a non-finite validation loss")
            totals.update(loss, components)
            progress.set_postfix(loss=f"{totals.loss / totals.batches:.2f}")
        return totals.finish()

    def _forward_loss(self, batch, sample: bool):
        image, target, valid = (
            tensor.to(self.device, non_blocking=True)
            for tensor in batch
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            mean, std = self.model(image)
            return self.loss_function(
                mean,
                std,
                target,
                valid,
                sample=sample,
            )

    def _optimizer_step(self, accumulated: int) -> None:
        self.scaler.unscale_(self.optimizer)
        if accumulated > 1:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)


class _MetricTotals:
    def __init__(self):
        self.loss = 0.0
        self.bce = 0.0
        self.kl = 0.0
        self.positive_pixels = 0.0
        self.batches = 0

    def update(self, loss: torch.Tensor, components: dict[str, torch.Tensor]):
        self.loss += float(loss.detach())
        self.bce += float(components["bce"])
        self.kl += float(components["kl"])
        self.positive_pixels += float(components["positive_pixels"])
        self.batches += 1

    def finish(self) -> EpochMetrics:
        if self.batches == 0:
            raise ValueError("DataLoader did not produce any batches")
        return EpochMetrics(
            loss=self.loss / self.batches,
            bce=self.bce / self.batches,
            kl=self.kl / self.batches,
            positive_pixels=self.positive_pixels / self.batches,
            batches=self.batches,
        )
