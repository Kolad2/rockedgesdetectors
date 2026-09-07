"""SGD parameter groups and learning-rate multipliers from upstream train.py."""

import torch


def create_rcf_optimizer(model, learning_rate=1e-6, momentum=0.9, weight_decay=2e-4):
    if learning_rate <= 0 or weight_decay < 0 or not 0 <= momentum < 1:
        raise ValueError("Invalid learning rate, weight decay or momentum")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer, kind = name.rsplit(".", 1)
        if layer == "score_fuse":
            family, scale = "fuse", 0.001
        elif layer.startswith("score_dsn"):
            family, scale = "side", 0.01
        elif layer.endswith("_down"):
            family, scale = "down", 0.1
        elif layer.startswith("conv5_"):
            family, scale = "conv5", 100.0
        elif layer.startswith(("conv1_", "conv2_", "conv3_", "conv4_")):
            family, scale = "conv1-4", 1.0
        else:
            raise ValueError(f"Unknown RCF parameter: {name}")
        if kind not in ("weight", "bias"):
            raise ValueError(f"Unknown RCF parameter: {name}")
        key = f"{family}.{kind}"
        if key not in groups:
            groups[key] = {
                "params": [], "name": key,
                "lr": learning_rate * scale * (2 if kind == "bias" else 1),
                "weight_decay": 0.0 if kind == "bias" else weight_decay,
            }
        groups[key]["params"].append(parameter)
    return torch.optim.SGD(list(groups.values()), lr=learning_rate, momentum=momentum)
