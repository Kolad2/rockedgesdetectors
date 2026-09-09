"""NumPy adapter for DDN edge inference."""

import torch

from ..numpy_adapter import NumpyAdapter
from .model import DDN


class NumpyDDNAdapter(NumpyAdapter):
    """Convert RGB images/batches to DDN edge probability maps."""

    def __init__(self, module: DDN, granularity: float = 0.0, normalize: bool = False):
        super().__init__(module)
        self.granularity = float(granularity)
        # Per-crop min-max scaling can create visible tile boundaries.
        self.normalize = bool(normalize)

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != 3:
            raise ValueError("DDN input must have 3 RGB channels")
        return tensor

    def _predict(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.module(tensor)
        edges = torch.sigmoid(mean + self.granularity * std)
        if self.normalize:
            minimum = edges.amin(dim=(-2, -1), keepdim=True)
            maximum = edges.amax(dim=(-2, -1), keepdim=True)
            edges = (edges - minimum) / (maximum - minimum).clamp_min(1e-8)
        return edges
