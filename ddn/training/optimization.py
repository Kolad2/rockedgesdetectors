"""Optimizer construction matching the official DDN parameter groups."""

import torch

from ..model import DDNBSDS


def create_ddn_optimizer(
    model: DDNBSDS,
    learning_rate: float = 1e-4,
    encoder_lr_scale: float = 0.1,
    weight_decay: float = 5e-4,
) -> torch.optim.Adam:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if encoder_lr_scale <= 0:
        raise ValueError("encoder_lr_scale must be positive")

    groups = {
        "pretrained.weight": [],
        "pretrained.bias": [],
        "new.weight": [],
        "new.bias": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_pretrained = name.startswith((
            "encoder.stages.",
            "encoder.downsample_layers.",
        ))
        kind = "weight" if "weight" in name else "bias"
        prefix = "pretrained" if is_pretrained else "new"
        groups[f"{prefix}.{kind}"].append(parameter)

    return torch.optim.Adam((
        {
            "params": groups["pretrained.weight"],
            "lr": learning_rate * encoder_lr_scale,
            "weight_decay": weight_decay,
        },
        {
            "params": groups["pretrained.bias"],
            "lr": learning_rate * 2.0 * encoder_lr_scale,
            "weight_decay": 0.0,
        },
        {
            "params": groups["new.weight"],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": groups["new.bias"],
            "lr": learning_rate * 2.0,
            "weight_decay": 0.0,
        },
    ))
