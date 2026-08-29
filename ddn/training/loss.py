"""Losses used by the original DDN training procedure."""

import torch
from torch import nn
from torch.nn import functional as F


class DDNLoss(nn.Module):
    """Class-balanced edge BCE plus Gaussian KL regularization."""

    def __init__(
        self,
        negative_weight: float = 1.1,
        kl_weight: float = 1e-2,
        variance_epsilon: float = 1e-8,
    ):
        super().__init__()
        if negative_weight <= 0:
            raise ValueError("negative_weight must be positive")
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        self.negative_weight = float(negative_weight)
        self.kl_weight = float(kl_weight)
        self.variance_epsilon = float(variance_epsilon)

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = valid.bool()

        target = target.to(dtype=mean.dtype)
        positive = (target >= 0.5) & valid
        negative = (target < 0.5) & valid
        positive_count = positive.sum().to(dtype=torch.float32)
        negative_count = negative.sum().to(dtype=torch.float32)
        valid_count = (positive_count + negative_count).clamp_min(1.0)

        weights = torch.zeros_like(target)
        weights[positive] = (negative_count / valid_count).to(weights.dtype)
        weights[negative] = (
            self.negative_weight * positive_count / valid_count
        ).to(weights.dtype)

        logits = mean + std * torch.randn_like(std) if sample else mean
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            weight=weights,
            reduction="sum",
        )

        mean_fp32 = mean.float()
        variance = std.float().square().clamp_min(self.variance_epsilon)
        kl_map = 0.5 * (
            mean_fp32.square() + variance - 1.0 - variance.log()
        )
        kl = kl_map.masked_select(valid).sum()
        total = bce.float() + self.kl_weight * kl
        return total, {
            "bce": bce.detach(),
            "kl": kl.detach(),
            "positive_pixels": positive_count.detach(),
            "valid_pixels": valid_count.detach(),
        }
