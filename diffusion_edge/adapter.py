"""NumPy image adapter for the fixed-size DiffusionEdge PyTorch module."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import DiffusionEdgeBSDS


class NumpyDiffusionEdgeAdapter(nn.Module):
    """Convert an RGB NumPy crop to a DiffusionEdge probability map."""

    def __init__(self, module: DiffusionEdgeBSDS):
        super().__init__()
        self.module = module

    def forward(self, image: np.ndarray | torch.Tensor):
        if isinstance(image, torch.Tensor):
            return self.module(image)
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray or torch.Tensor, "
                f"got {type(image).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy image must have shape [height, width, 3]")

        height, width = image.shape[:2]
        input_size = self.module.input_size
        if height > input_size or width > input_size:
            raise ValueError(
                "NumpyDiffusionEdgeAdapter accepts one crop no larger than "
                f"{input_size}x{input_size}; use Cropper for larger images"
            )

        array = np.ascontiguousarray(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.module.device, dtype=torch.float32)
        if image.dtype == np.uint8:
            tensor = tensor / 255.0
        elif tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
            raise ValueError("Floating-point NumPy images must be in the [0, 1] range")

        pad_right = input_size - width
        pad_bottom = input_size - height
        if pad_right or pad_bottom:
            tensor = F.pad(
                tensor,
                (0, pad_right, 0, pad_bottom),
                mode="replicate",
            )

        with torch.inference_mode():
            output = self.module(tensor)
        return output[0, 0, :height, :width].detach().cpu().numpy()
