"""Training orchestration; entry scripts only need settings and run_training."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..model_rcf import RCFBSDS
from .checkpoint import restore_training_checkpoint, save_training_checkpoint
from .data import EdgeManifestDataset
from .loss import RCFLoss
from .optimization import create_rcf_optimizer
from .trainer import RCFTrainer


@dataclass(frozen=True)
class TrainingConfig:
    train_manifest: Path
    initial_checkpoint: Path
    checkpoint_folder: Path
    resume_checkpoint: Path | None = None
    epochs: int = 10
    crop_size: int = 320
    batch_size: int = 1
    accumulation_steps: int = 10
    num_workers: int = 2
    validation_fraction: float = 0.1
    seed: int = 42
    learning_rate: float = 1e-6
    momentum: float = 0.9
    weight_decay: float = 2e-4
    lr_step_size: int = 3
    lr_gamma: float = 0.1
    negative_pixel_weight: float = 1.1
    label_threshold: float = 0.5
    ignore_ambiguous: bool = True
    min_edge_pixels_per_crop: int = 5
    crop_attempts: int = 10
    gradient_clip_norm: float | None = None
    augment_flips: bool = True
    save_every_epochs: int = 1
    device: str = "cuda"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "accumulation_steps", "lr_step_size",
                     "save_every_epochs", "crop_attempts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.crop_size < 9 or self.num_workers < 0:
            raise ValueError("crop_size must be >= 9 and num_workers >= 0")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.lr_gamma <= 0:
            raise ValueError("lr_gamma must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(length, validation_fraction, seed):
    if length < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Need nonempty dataset and validation_fraction in [0, 1)")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    size = int(round(length * validation_fraction))
    if length == 1:
        size = 0
    elif validation_fraction > 0:
        size = max(1, min(size, length - 1))
    return indices[size:], indices[:size]


def create_loaders(config):
    common_dataset = dict(
        manifest_path=config.train_manifest, crop_size=config.crop_size,
        label_threshold=config.label_threshold,
        ignore_ambiguous=config.ignore_ambiguous,
    )
    train = EdgeManifestDataset(
        **common_dataset, crop_mode="random", augment=config.augment_flips,
        min_edge_pixels=config.min_edge_pixels_per_crop, crop_attempts=config.crop_attempts,
    )
    validation = EdgeManifestDataset(
        **common_dataset, crop_mode="center", augment=False, min_edge_pixels=0,
    )
    train_indices, validation_indices = split_indices(
        len(train), config.validation_fraction, config.seed,
    )
    common_loader = dict(
        batch_size=config.batch_size, num_workers=config.num_workers,
        pin_memory=torch.device(config.device).type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(Subset(train, train_indices), shuffle=True, **common_loader)
    validation_loader = (
        DataLoader(Subset(validation, validation_indices), shuffle=False, **common_loader)
        if validation_indices else None
    )
    print(f"RCF dataset: train={len(train_indices)}, validation={len(validation_indices)}")
    return train_loader, validation_loader


def print_metrics(epoch, split, metrics):
    print(f"epoch={epoch} split={split} loss={metrics.loss:.4f} "
          f"fused_bce={metrics.fused_bce:.4f} positive_pixels={metrics.positive_pixels:.1f}")


def run_training(config: TrainingConfig) -> None:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RCF training is configured for CUDA but CUDA is unavailable")
    seed_everything(config.seed)
    train_loader, validation_loader = create_loaders(config)
    model = RCFBSDS(
        config.resume_checkpoint or config.initial_checkpoint, trainable=True,
    ).to(device)
    optimizer = create_rcf_optimizer(
        model, config.learning_rate, config.momentum, config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma,
    )
    trainer = RCFTrainer(
        model, optimizer, RCFLoss(config.negative_pixel_weight), device,
        accumulation_steps=config.accumulation_steps,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    start_epoch = 1
    if config.resume_checkpoint is not None:
        completed_epoch = restore_training_checkpoint(
            config.resume_checkpoint, model, optimizer, scheduler=scheduler,
            map_location=device,
        )
        start_epoch = completed_epoch + 1
        print(f"Resumed RCF training after epoch {completed_epoch}")
    for epoch in range(start_epoch, config.epochs + 1):
        training = trainer.train_epoch(train_loader, epoch)
        validation = (
            trainer.validate(validation_loader, epoch)
            if validation_loader is not None else None
        )
        scheduler.step()
        print_metrics(epoch, "train", training)
        if validation is not None:
            print_metrics(epoch, "validation", validation)
        if epoch % config.save_every_epochs == 0 or epoch == config.epochs:
            path = save_training_checkpoint(
                Path(config.checkpoint_folder) / f"checkpoint_{epoch:03d}.pth",
                epoch, model, optimizer, scheduler=scheduler,
                metrics={"train": asdict(training),
                         "validation": asdict(validation) if validation is not None else None},
            )
            print(f"Saved RCF checkpoint: {path}")
