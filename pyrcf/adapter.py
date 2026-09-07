"""RGB NumPy images to RCF fused edge probabilities."""

import numpy as np
import torch
from torch import nn

from .model_rcf import RCFBSDS


class NumpyRCFAdapter(nn.Module):
    """Accept RGB uint8 [0,255] or floating point [0,1], return an HxW map."""

    def __init__(self, module: RCFBSDS):
        super().__init__()
        self.module = module

    @torch.inference_mode()
    def forward(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 9:
            raise ValueError("image must have shape [H, W, 3] with H and W >= 9")
        if image.dtype == np.uint8:
            array = image.astype(np.float32)
        elif np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Floating-point images must be finite and in [0, 1]")
            array = image.astype(np.float32) * 255.0
        else:
            raise TypeError("image must have uint8 or floating-point dtype")
        # Match upstream dataset.py: OpenCV BGR pixels minus the training mean.
        mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        array = np.ascontiguousarray((array[..., ::-1] - mean).transpose(2, 0, 1))
        parameter = next(self.module.parameters())
        tensor = torch.from_numpy(array).unsqueeze(0).to(parameter)
        return self.module(tensor)[-1][0, 0].float().cpu().numpy()
