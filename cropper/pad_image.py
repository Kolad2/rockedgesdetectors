"""Pad images so overlap-tile counts are integral along both axes."""

import math

import numpy as np


def pad_image(
    image: np.ndarray,
    crop: int = 512,
    pad: int = 64,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D NumPy array")
    if crop <= 0:
        raise ValueError("crop must be positive")
    if pad < 0 or crop <= 2 * pad:
        raise ValueError("crop must be greater than 2 * pad >= 0")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    step = crop - 2 * pad
    tiles_y = math.ceil(height / step)
    tiles_x = math.ceil(width / step)
    padded_height = tiles_y * step + 2 * pad
    padded_width = tiles_x * step + 2 * pad

    padding_y = padded_height - height
    padding_x = padded_width - width
    top = padding_y // 2
    bottom = padding_y - top
    left = padding_x // 2
    right = padding_x - left

    padding = [(top, bottom), (left, right)]
    if image.ndim == 3:
        padding.append((0, 0))
    padded_image = np.pad(
        image,
        pad_width=tuple(padding),
        mode="constant",
        constant_values=0,
    )
    return padded_image, (top, bottom, left, right)
