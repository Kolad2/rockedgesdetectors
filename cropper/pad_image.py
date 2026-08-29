"""Pad images so overlap-tile counts are integral along both axes."""

import math

import numpy as np


PAD_MODES = (
    "reflect",
    "zeros",
    "constant",
    "replicate",
    "edge",
    "circular",
    "wrap",
    "symmetric",
)

_PAD_MODE_ALIASES = {
    "zeros": "constant",
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
    "edge": "edge",
    "circular": "wrap",
    "wrap": "wrap",
    "symmetric": "symmetric",
}


def resolve_pad_mode(pad_mode: str) -> str:
    """Translate common PyTorch/NumPy padding names to a NumPy mode."""
    if not isinstance(pad_mode, str):
        raise TypeError("pad_mode must be a string")

    normalized = pad_mode.strip().lower()
    try:
        return _PAD_MODE_ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(PAD_MODES)
        raise ValueError(
            f"Unsupported pad_mode {pad_mode!r}. Supported modes: {supported}"
        ) from error


def pad_image(
    image: np.ndarray,
    crop: int = 512,
    pad: int = 64,
    pad_mode: str = "reflect",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D NumPy array")
    if crop <= 0:
        raise ValueError("crop must be positive")
    if pad < 0 or crop <= 2 * pad:
        raise ValueError("crop must be greater than 2 * pad >= 0")
    numpy_pad_mode = resolve_pad_mode(pad_mode)

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
    pad_kwargs = {"constant_values": 0} if numpy_pad_mode == "constant" else {}
    padded_image = np.pad(
        image,
        pad_width=tuple(padding),
        mode=numpy_pad_mode,
        **pad_kwargs,
    )
    return padded_image, (top, bottom, left, right)
