"""RGB NumPy images/batches to RCF fused edge probabilities."""

import torch

from ..numpy_adapter import NumpyAdapter


class NumpyRCFAdapter(NumpyAdapter):
    """Apply RCF BGR preprocessing and select the fused edge map."""

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != 3 or min(tensor.shape[-2:]) < 9:
            raise ValueError("RCF input must have 3 RGB channels and H, W >= 9")
        # Match upstream dataset.py: BGR pixels in [0, 255] minus training mean.
        mean = tensor.new_tensor([104.00698793, 116.66876762, 122.67891434])[None, :, None, None]
        return tensor[:, [2, 1, 0]] * 255.0 - mean

    def _predict(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.module(tensor)[-1]
