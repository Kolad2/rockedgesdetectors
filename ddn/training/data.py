"""Dataset utilities for fine-tuning DDN on image/edge manifest files."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass(frozen=True)
class EdgeSample:
    image: Path
    label: Path
    mask: Path | None = None


def load_edge_manifest(path: str | Path) -> list[EdgeSample]:
    """Read tab-separated image/label[/mask] paths from a manifest."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    samples: list[EdgeSample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) == 1:
                row = row[0].split()
            if len(row) not in (2, 3):
                raise ValueError(
                    f"{manifest_path}, line {line_number}: expected "
                    f"image/label[/mask], got {len(row)} columns"
                )

            values = [value.strip() for value in row]
            image_path = _resolve_path(manifest_path.parent, values[0])
            label_path = _resolve_path(manifest_path.parent, values[1])
            mask_path = (
                _resolve_path(manifest_path.parent, values[2])
                if len(values) == 3 and values[2]
                else None
            )

            for field_name, sample_path in (
                ("image", image_path),
                ("label", label_path),
                ("mask", mask_path),
            ):
                if sample_path is not None and not sample_path.is_file():
                    raise FileNotFoundError(
                        f"{field_name} from line {line_number} not found: "
                        f"{sample_path}"
                    )
            samples.append(EdgeSample(image_path, label_path, mask_path))

    if not samples:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")
    return samples


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


class EdgeManifestDataset(Dataset):
    """RGB images and binary edge maps prepared for DDN."""

    def __init__(
        self,
        manifest_path: str | Path,
        crop_size: int | None = 320,
        crop_mode: str = "random",
        label_threshold: float = 0.5,
        augment: bool = False,
        min_edge_pixels: int = 1,
        crop_attempts: int = 10,
    ):
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive or None")
        if crop_mode not in ("random", "center"):
            raise ValueError("crop_mode must be 'random' or 'center'")
        if not 0 <= label_threshold <= 1:
            raise ValueError("label_threshold must be in [0, 1]")
        if min_edge_pixels < 0:
            raise ValueError("min_edge_pixels must be non-negative")
        if crop_attempts <= 0:
            raise ValueError("crop_attempts must be positive")

        self.samples = load_edge_manifest(manifest_path)
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.label_threshold = label_threshold
        self.augment = augment
        self.min_edge_pixels = min_edge_pixels
        self.crop_attempts = crop_attempts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image) as image_file:
            image = pil_to_tensor(image_file.convert("RGB")).float() / 255.0
        with Image.open(sample.label) as label_file:
            label = pil_to_tensor(label_file.convert("L")).float() / 255.0

        if image.shape[-2:] != label.shape[-2:]:
            raise ValueError(
                f"Image and label sizes differ for {sample.image}: "
                f"{tuple(image.shape[-2:])} != {tuple(label.shape[-2:])}"
            )

        target = (label >= self.label_threshold).float()
        if sample.mask is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            with Image.open(sample.mask) as mask_file:
                mask = pil_to_tensor(mask_file.convert("L")).float() / 255.0
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"Image and mask sizes differ for {sample.image}: "
                    f"{tuple(image.shape[-2:])} != {tuple(mask.shape[-2:])}"
                )
            valid = mask >= self.label_threshold

        if self.crop_size is not None:
            image, target, valid = self._crop(image, target, valid)
        if self.augment:
            image, target, valid = self._augment(image, target, valid)
        return image, target, valid

    def _crop(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.crop_size
        image, target, valid = _pad_to_size(image, target, valid, size)
        height, width = image.shape[-2:]

        if self.crop_mode == "center":
            top = (height - size) // 2
            left = (width - size) // 2
            return _take_crop(image, target, valid, top, left, size)

        best = None
        best_edge_count = -1
        for _ in range(self.crop_attempts):
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            candidate = _take_crop(image, target, valid, top, left, size)
            edge_count = torch.count_nonzero(candidate[1] * candidate[2]).item()
            if edge_count > best_edge_count:
                best = candidate
                best_edge_count = edge_count
            if edge_count >= self.min_edge_pixels:
                return candidate
        return best

    @staticmethod
    def _augment(
        image: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
            valid = valid.flip(-1)
        if random.random() < 0.5:
            image = image.flip(-2)
            target = target.flip(-2)
            valid = valid.flip(-2)
        return image, target, valid


def _pad_to_size(
    image: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    pad_bottom = max(0, size - height)
    pad_right = max(0, size - width)
    if not pad_bottom and not pad_right:
        return image, target, valid

    padding = (0, pad_right, 0, pad_bottom)
    image = F.pad(image, padding, mode="replicate")
    target = F.pad(target, padding, value=0.0)
    valid = F.pad(valid, padding, value=False)
    return image, target, valid


def _take_crop(
    image: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    top: int,
    left: int,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = slice(top, top + size)
    columns = slice(left, left + size)
    return (
        image[:, rows, columns],
        target[:, rows, columns],
        valid[:, rows, columns],
    )
