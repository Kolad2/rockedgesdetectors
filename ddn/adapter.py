"""NumPy adapter for DDN edge inference."""

import numpy as np
import torch
from torch import nn

from .model import DDNBSDS


class NumpyDDNAdapter(nn.Module):
    """Convert an RGB NumPy image to a DDN edge probability map."""

    def __init__(
        self,
        module: DDNBSDS,
        granularity: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        self.module = module
        self.granularity = float(granularity)
        # Leave normalization disabled when this adapter is used by Cropper.
        # Per-crop min-max scaling creates visible tile boundaries.
        self.normalize = bool(normalize)

    def _predict(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.module(tensor)
        edges = torch.sigmoid(mean + self.granularity * std)
        if self.normalize:
            minimum = edges.amin(dim=(-2, -1), keepdim=True)
            maximum = edges.amax(dim=(-2, -1), keepdim=True)
            edges = (edges - minimum) / (maximum - minimum).clamp_min(1e-8)
        return edges

    def forward(self, image: np.ndarray | torch.Tensor):
        if isinstance(image, torch.Tensor):
            return self._predict(image)
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray or torch.Tensor, "
                f"got {type(image).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy image must have shape [height, width, 3]")

        array = np.ascontiguousarray(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.module.device, dtype=torch.float32)
        if image.dtype == np.uint8:
            tensor = tensor / 255.0
        elif tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
            raise ValueError("Floating-point NumPy images must be in the [0, 1] range")

        with torch.inference_mode():
            edges = self._predict(tensor)
        return edges[0, 0].detach().cpu().numpy()
