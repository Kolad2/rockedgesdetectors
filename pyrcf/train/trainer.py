"""Adaptation of the float32 train loop in yun-liu/RCF-PyTorch/train.py."""

from dataclasses import dataclass
import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    fused_bce: float
    positive_pixels: float
    batches: int


class RCFTrainer:
    def __init__(self, model, optimizer, loss_function, device,
                 accumulation_steps=10, gradient_clip_norm=None):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.iter_size = accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm

    def train_epoch(self, loader, epoch):
        return self._run_epoch(loader, epoch, training=True)

    @torch.no_grad()
    def validate(self, loader, epoch):
        return self._run_epoch(loader, epoch, training=False)

    def _run_epoch(self, loader, epoch, training):
        if len(loader) == 0:
            raise ValueError("DataLoader did not produce any batches")
        self.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        total_loss = total_fused = total_positive = 0.0
        split = "train" if training else "validation"
        progress = tqdm(loader, desc=f"RCF {split} {epoch}")
        for index, (image, label) in enumerate(progress):
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            outputs = self.model(image)
            loss, parts = self.loss_function(outputs, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite RCF {split} loss")
            if training:
                # Authors divide the sum of six losses by iter_size before
                # backward. Also flush and correctly scale a final short group.
                group_start = (index // self.iter_size) * self.iter_size
                group_size = min(self.iter_size, len(loader) - group_start)
                (loss / group_size).backward()
                if (index + 1) % self.iter_size == 0 or index + 1 == len(loader):
                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            total_loss += loss.detach().item()
            total_fused += parts["fused_bce"].item()
            total_positive += parts["positive_pixels"].item()
            progress.set_postfix(loss=f"{total_loss / (index + 1):.3f}")
        count = len(loader)
        return EpochMetrics(total_loss / count, total_fused / count,
                            total_positive / count, count)
