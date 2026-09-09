"""Shared NumPy image/batch interface for PyTorch edge detectors."""

from typing import Any

import numpy as np
import torch
from torch import nn


class NumpyAdapter(nn.Module):
    """Adapt HWC images or NHWC batches to a PyTorch edge detector.

    NumPy inputs must be uint8 [0, 255] or finite floats [0, 1]. They
    become NCHW tensors in [0, 1] on the module's device and floating dtype.
    Results are float32 NumPy maps: HW for an image, NHW for a batch.

    Override preprocess for network-specific normalization/padding, _predict
    for network invocation/output selection, and postprocess for restoring
    the original spatial size. Hooks operate on the entire batch.

    NumPy inference disables gradients but does not change training mode;
    call adapter.eval() before inference. Tensor inputs are already prepared:
    they go directly to _predict, preserving device, dtype and autograd.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module

    def _reference_tensor(self) -> torch.Tensor | None:
        parameter = next(self.module.parameters(), None)
        return parameter if parameter is not None else next(self.module.buffers(), None)

    @property
    def device(self) -> torch.device:
        reference = self._reference_tensor()
        return reference.device if reference is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        reference = self._reference_tensor()
        if reference is not None and reference.is_floating_point():
            return reference.dtype
        return torch.float32

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Receive NCHW values in [0, 1]; return network-ready input."""
        return tensor

    def _predict(self, tensor: torch.Tensor) -> Any:
        """Run the network; override to select or calculate edge maps."""
        return self.module(tensor)

    def postprocess(self, output: Any, image_size: tuple[int, int]) -> torch.Tensor:
        """Return NHW or N1HW edge maps at the original (height, width)."""
        return output

    def forward(self, image: np.ndarray | torch.Tensor) -> Any:
        if isinstance(image, torch.Tensor):
            return self._predict(image)
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray or torch.Tensor")
        if image.ndim not in (3, 4):
            raise ValueError("NumPy input must have shape [H, W, C] or [N, H, W, C]")
        if any(size == 0 for size in image.shape):
            raise ValueError("NumPy input dimensions must be positive")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.floating):
                raise TypeError("NumPy input must have uint8 or floating-point dtype")
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Floating-point NumPy inputs must be finite and in [0, 1]")

        single_image = image.ndim == 3
        batch = image[np.newaxis] if single_image else image
        image_size = tuple(batch.shape[1:3])
        # Copy only when needed for negative strides, read-only arrays, or
        # non-native NumPy floating dtypes unsupported by torch.from_numpy.
        array = batch
        if array.dtype != np.uint8 and array.dtype not in (np.float16, np.float32, np.float64):
            array = array.astype(np.float32)
        array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            array = array.copy()
        with torch.inference_mode():
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
            tensor = tensor.to(device=self.device, dtype=self.dtype)
            if image.dtype == np.uint8:
                tensor = tensor / 255.0
            tensor = self.preprocess(tensor)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("preprocess must return a torch.Tensor")
            output = self.postprocess(self._predict(tensor), image_size)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Adapter must produce a torch.Tensor of edge maps")
            if output.ndim == 4 and output.shape[1] == 1:
                output = output[:, 0]
            expected_shape = (len(batch), *image_size)
            if tuple(output.shape) != expected_shape:
                raise ValueError(
                    f"Adapter returned shape {tuple(output.shape)}, expected {expected_shape}"
                )
            result = output.detach().to(device="cpu", dtype=torch.float32).numpy()
        return result[0] if single_image else result
