"""Original RCF class-balanced BCE and deep supervision (upstream utils.py)."""

import torch
from torch import nn
from torch.nn import functional as F


class RCFLoss(nn.Module):
    """Sum pixel BCE over five side outputs and fusion, all equally weighted.

    Targets are 0 (background), 1 (edge), or 2 (ignored). An optional boolean
    valid mask also excludes pixels. Balancing counts cover the whole batch,
    as in upstream; single-class and completely ignored batches give zero loss.
    """

    def __init__(self, negative_weight: float = 1.1,
                 reference_pixels: int = 320 * 320):
        super().__init__()
        if negative_weight <= 0 or reference_pixels <= 0:
            raise ValueError("negative_weight and reference_pixels must be positive")
        self.negative_weight = float(negative_weight)
        self.reference_pixels = int(reference_pixels)

    def forward(self, outputs, target, valid=None):
        if len(outputs) != 6:
            raise ValueError("RCF loss expects five side outputs and one fused output")
        if valid is not None and valid.shape != target.shape:
            raise ValueError("valid and target shapes must match")
        selected = target != 2
        if valid is not None:
            selected = selected & valid.bool()
        labels = target[selected].float()
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Valid labels must be 0 or 1 (2 means ignored)")
        positive = (labels == 1).sum().float()
        negative = (labels == 0).sum().float()
        count = (positive + negative).clamp_min(1)
        weights = torch.where(labels == 1, negative / count,
                              self.negative_weight * positive / count)
        losses = []
        for prediction in outputs:
            if prediction.shape != target.shape:
                raise ValueError("RCF output and target shapes must match")
            losses.append(F.binary_cross_entropy(
                prediction[selected].float(), labels, weight=weights, reduction="sum",
            ))
        # Upstream sums over pixels, making a 1024 crop produce about 10.24x
        # the gradient of a 320 crop. Normalize to a 320x320-equivalent area
        # while retaining the original loss/LR scale at that reference size.
        normalization = self.reference_pixels * target.shape[0] / count
        losses = [loss * normalization for loss in losses]
        total = torch.stack(losses).sum()
        return total, {
            "bce": total.detach(),
            "fused_bce": losses[-1].detach(),
            "positive_pixels": positive.detach(),
        }
