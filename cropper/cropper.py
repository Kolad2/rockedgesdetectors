"""Overlap-tile inference with configurable symmetric image padding."""

import operator

import numpy as np

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
    """Pass batches of NumPy crops to a callable model or NumPy adapter.

    For batch_num > 1, RGB/multichannel inputs produce [N, H, W, C]
    batches; grayscale inputs produce [N, H, W, 1]. The callable must
    return NumPy edge maps [N, H, W], including for a final batch of one.
    Input dtype and values are preserved. Tensor conversion, normalization,
    device placement and output selection belong to the adapter.

    batch_num=1 uses the standard Cropper path with individual HW/HWC crops.
    Images and assembled float32 output maps remain in CPU memory.
    """

    def __init__(
        self,
        model,
        crop: int = 512,
        pad: int = 64,
        pad_mode: str = "reflect",
        display: bool = False,
        batch_num: int = 1,
    ):
        super().__init__(model, crop, pad, pad_mode, display)
        self.batch_num = operator.index(batch_num)
        if self.batch_num < 1:
            raise ValueError("batch_num must be positive")
        if not callable(model):
            raise TypeError("BatchedCropper model must be callable")

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
        if crops.ndim == 3:
            crops = crops[..., np.newaxis]
        outputs = self.model(crops)
        if not isinstance(outputs, np.ndarray):
            raise TypeError("BatchedCropper model must return a numpy.ndarray")
        expected_shape = (len(coordinates), self.crop, self.crop)
        if outputs.shape != expected_shape:
            raise ValueError(f"Model returned shape {outputs.shape}, expected {expected_shape}")
        for (x, y), output_crop in zip(coordinates, outputs):
            self.output[
                y + self.pad:y + self.pad + self.step,
                x + self.pad:x + self.pad + self.step,
            ] = output_crop[self.pad:self.crop - self.pad, self.pad:self.crop - self.pad]
        if self.progress is not None:
            self.progress.update(len(coordinates))
