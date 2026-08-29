"""Overlap-tile inference with symmetric black image padding."""

import numpy as np

from .pad_image import pad_image


class Cropper:
    def __init__(
        self,
        model,
        crop: int = 512,
        pad: int = 64,
        display: bool = False,
    ):
        self.crop = int(crop)
        self.pad = int(pad)
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

        output_crop = self.model(image_crop)
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
