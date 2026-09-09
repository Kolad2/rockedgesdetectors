"""ImageNet normalization and output selection for NumPy edge inference."""

from typing import Any, Callable, Optional

import torch
from torch import nn

from .numpy_adapter import NumpyAdapter


ModelOutput = Any
OutputSelector = Callable[[ModelOutput], torch.Tensor]


def _identity(output: ModelOutput) -> torch.Tensor:
    return output


class NumpyImagenetAdapter(NumpyAdapter):
    """Accept RGB images/batches and apply ImageNet normalization."""

    def __init__(self, module: nn.Module, output_selector: Optional[OutputSelector] = None):
        super().__init__(module)
        self.output_selector = output_selector or _identity

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != 3:
            raise ValueError("ImageNet input must have 3 RGB channels")
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (tensor - mean) / std

    def postprocess(self, output: ModelOutput, image_size: tuple[int, int]) -> torch.Tensor:
        # Tensor inputs retain their previous behavior: return raw network outputs.
        return self.output_selector(output)
