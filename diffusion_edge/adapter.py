"""NumPy adapter for the fixed-size DiffusionEdge module."""

import torch
from torch.nn import functional as F

from ..numpy_adapter import NumpyAdapter


class NumpyDiffusionEdgeAdapter(NumpyAdapter):
    """Pad RGB crops to the network size and trim predicted maps back."""

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != 3:
            raise ValueError("DiffusionEdge input must have 3 RGB channels")
        height, width = tensor.shape[-2:]
        input_size = self.module.input_size
        if height > input_size or width > input_size:
            raise ValueError(
                f"DiffusionEdge crops must be no larger than {input_size}x{input_size}; "
                "use Cropper for larger images"
            )
        if height != input_size or width != input_size:
            tensor = F.pad(tensor, (0, input_size - width, 0, input_size - height), mode="replicate")
        return tensor

    def postprocess(self, output: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        return output[..., :height, :width]
