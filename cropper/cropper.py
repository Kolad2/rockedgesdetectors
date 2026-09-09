"""Overlap-tile inference with configurable symmetric image padding."""

import operator
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .pad_image import PAD_MODES, pad_image, resolve_pad_mode


class Cropper:
    """Run overlap-tile inference.

    Available padding modes are exposed as ``Cropper.PAD_MODES``.
    """

    PAD_MODES = PAD_MODES

    def __init__(
        self,
        model,
        crop: int = 512,
        pad: int = 64,
        pad_mode: str = "reflect",
        display: bool = False,
    ):
        self.crop = int(crop)
        self.pad = int(pad)
        self.pad_mode = (
            pad_mode.strip().lower()
            if isinstance(pad_mode, str)
            else pad_mode
        )
        self.display = display
        self.model = model

        self.image = None
        self.output = None
        self.progress = None
        self.sh = None

        if self.crop <= 0:
            raise ValueError("crop must be positive")
        if self.pad < 0:
            raise ValueError("pad must be non-negative")
        if self.crop <= 2 * self.pad:
            raise ValueError("crop must be greater than 2 * pad")
        resolve_pad_mode(self.pad_mode)

    @property
    def step(self) -> int:
        return self.crop - 2 * self.pad

    def get_crop_edge(self, x: int, y: int) -> None:
        image_crop = self.image[
            y:y + self.crop,
            x:x + self.crop,
        ]
        if image_crop.shape[:2] != (self.crop, self.crop):
            raise RuntimeError(
                f"Internal crop has shape {image_crop.shape[:2]}, "
                f"expected {(self.crop, self.crop)}"
            )

        output_crop = self._predict_crop(image_crop)
        if not isinstance(output_crop, np.ndarray):
            raise TypeError(
                "Cropper model must return a numpy.ndarray, "
                f"got {type(output_crop).__name__}"
            )
        if output_crop.shape != (self.crop, self.crop):
            raise ValueError(
                f"Model returned shape {output_crop.shape}, "
                f"expected {(self.crop, self.crop)}"
            )

        output_y = y + self.pad
        output_x = x + self.pad
        valid_size = self.step
        self.output[
            output_y:output_y + valid_size,
            output_x:output_x + valid_size,
        ] = output_crop[
            self.pad:self.crop - self.pad,
            self.pad:self.crop - self.pad,
        ]

        if self.progress is not None:
            self.progress.update(1)

    def _predict_crop(self, image_crop: np.ndarray) -> np.ndarray:
        return self.model(image_crop)

    def center_edges(self) -> None:
        tiles_y, tiles_x = self._tile_counts()
        for tile_y in range(tiles_y):
            y = tile_y * self.step
            for tile_x in range(tiles_x):
                x = tile_x * self.step
                self.get_crop_edge(x, y)

    def _tile_counts(self) -> tuple[int, int]:
        tiles_y = (self.sh[0] - 2 * self.pad) // self.step
        tiles_x = (self.sh[1] - 2 * self.pad) // self.step
        return tiles_y, tiles_x

    def _count_crops(self) -> int:
        tiles_y, tiles_x = self._tile_counts()
        return tiles_y * tiles_x

    @staticmethod
    def _make_progress(
        total: int,
        enabled: bool,
        desc: str = "crop inference",
    ):
        if not enabled:
            return None
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("tqdm is not installed; progress display disabled")
            return None
        return tqdm(total=total, desc=desc)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray, "
                f"got {type(image).__name__}"
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                "image must have shape [height, width] or "
                "[height, width, channels]"
            )
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image height and width must be positive")

        original_height, original_width = image.shape[:2]
        padded_image, padding = pad_image(
            image,
            crop=self.crop,
            pad=self.pad,
            pad_mode=self.pad_mode,
        )
        top, _bottom, left, _right = padding

        self.image = padded_image
        self.sh = padded_image.shape
        self.output = np.zeros(padded_image.shape[:2], dtype=np.float32)
        self.progress = self._make_progress(
            total=self._count_crops(),
            enabled=self.display,
        )
        try:
            self.center_edges()
        finally:
            if self.progress is not None:
                self.progress.close()
                self.progress = None

        return self.output[
            top:top + original_height,
            left:left + original_width,
        ].copy()


class BatchedCropper(Cropper):
    """Run a PyTorch module on batches of NumPy image crops.

    Inputs become NCHW tensors on the model's device and floating dtype
    (CPU/float32 for stateless modules). uint8 values are divided by 255;
    floating image values are preserved. Optional preprocess receives this
    tensor batch, e.g. for ImageNet normalization. Optional output_selector
    converts the network output to [N, 1, H, W] or [N, H, W] edge maps.

    The module is set to eval mode and runs under torch.inference_mode().
    batch_num=1 uses Cropper's standard traversal and assembly, with tensor
    inference for each crop. The returned image is a float32 NumPy array.
    """

    def __init__(
        self,
        model: nn.Module,
        crop: int = 512,
        pad: int = 64,
        pad_mode: str = "reflect",
        display: bool = False,
        batch_num: int = 1,
        *,
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        output_selector: Callable[[Any], torch.Tensor] | None = None,
    ):
        super().__init__(model, crop, pad, pad_mode, display)
        self.batch_num = operator.index(batch_num)
        if self.batch_num < 1:
            raise ValueError("batch_num must be positive")
        if not isinstance(model, nn.Module):
            raise TypeError("BatchedCropper model must be a torch.nn.Module")
        self.preprocess = preprocess
        self.output_selector = output_selector
        self.model.eval()

    def _predict_crop(self, image_crop: np.ndarray) -> np.ndarray:
        return self._predict_batch(image_crop[np.newaxis])[0]

    def _predict_batch(self, crops: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(crops))
        tensor = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor.permute(0, 3, 1, 2)
        reference = next(self.model.parameters(), None)
        if reference is None:
            reference = next(self.model.buffers(), None)
        device = reference.device if reference is not None else torch.device("cpu")
        dtype = reference.dtype if reference is not None and reference.is_floating_point() else torch.float32
        with torch.inference_mode():
            tensor = tensor.to(device=device, dtype=dtype)
            if crops.dtype == np.uint8:
                tensor = tensor / 255.0
            if self.preprocess is not None:
                tensor = self.preprocess(tensor)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("preprocess must return a torch.Tensor")
            outputs = self.model(tensor)
            if self.output_selector is not None:
                outputs = self.output_selector(outputs)
            if not isinstance(outputs, torch.Tensor):
                raise TypeError("Model output must be a torch.Tensor; provide output_selector for structured outputs")
            if outputs.ndim == 4 and outputs.shape[1] == 1:
                outputs = outputs[:, 0]
            expected_shape = (len(crops), self.crop, self.crop)
            if tuple(outputs.shape) != expected_shape:
                raise ValueError(f"Model returned shape {tuple(outputs.shape)}, expected {expected_shape}")
            return outputs.detach().to(device="cpu", dtype=torch.float32).numpy()

    def center_edges(self) -> None:
        if self.batch_num == 1:
            return super().center_edges()

        tiles_y, tiles_x = self._tile_counts()
        coordinates = []
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                coordinates.append((tile_x * self.step, tile_y * self.step))
                if len(coordinates) == self.batch_num:
                    self._get_batch_edges(coordinates)
                    coordinates.clear()
        if coordinates:
            self._get_batch_edges(coordinates)

    def _get_batch_edges(self, coordinates: list[tuple[int, int]]) -> None:
        crops = np.stack([
            self.image[y:y + self.crop, x:x + self.crop]
            for x, y in coordinates
        ])
        outputs = self._predict_batch(crops)
        for (x, y), output_crop in zip(coordinates, outputs):
            self.output[
                y + self.pad:y + self.pad + self.step,
                x + self.pad:x + self.pad + self.step,
            ] = output_crop[self.pad:self.crop - self.pad, self.pad:self.crop - self.pad]
        if self.progress is not None:
            self.progress.update(len(coordinates))
