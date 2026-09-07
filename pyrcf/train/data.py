"""Adapt upstream dataset.py to image/label[/mask] manifests and project crops."""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


BGR_MEAN = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)


def load_edge_manifest(path):
    path = Path(path).resolve()
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) == 1:
                row = row[0].split()
            if len(row) not in (2, 3) or not all(value.strip() for value in row[:2]):
                raise ValueError(f"{path}:{line}: expected image label [mask]")
            paths = []
            for value in row:
                candidate = Path(value.strip()) if value.strip() else None
                if candidate is not None:
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    if not candidate.is_file():
                        raise FileNotFoundError(f"{path}:{line}: {candidate}")
                paths.append(candidate)
            samples.append(tuple(paths + [None] * (3 - len(paths))))
    if not samples:
        raise ValueError(f"Empty manifest: {path}")
    return samples


def _read_image(path, flags):
    # imdecode also supports non-ASCII Windows paths.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


class EdgeManifestDataset(Dataset):
    """Upstream BGR input and labels 0/1/2 (background/edge/ignore).

    By default, nonzero labels below threshold are ignored, as in BSDS_Dataset.
    Set ignore_ambiguous=False only for binary thresholding of project labels.
    An optional third manifest column masks out pixels; padding is ignored too.
    """

    def __init__(self, manifest_path, crop_size=320, crop_mode="random",
                 label_threshold=0.5, augment=False, min_edge_pixels=5,
                 crop_attempts=10, ignore_ambiguous=True):
        if crop_size is not None and crop_size < 9:
            raise ValueError("crop_size must be >= 9 or None")
        if crop_mode not in ("random", "center"):
            raise ValueError("crop_mode must be random or center")
        if not 0 < label_threshold <= 1 or min_edge_pixels < 0 or crop_attempts < 1:
            raise ValueError("Invalid label threshold or crop settings")
        self.samples = load_edge_manifest(manifest_path)
        self.crop_size, self.crop_mode = crop_size, crop_mode
        self.label_threshold = label_threshold
        self.augment = augment
        self.min_edge_pixels, self.crop_attempts = min_edge_pixels, crop_attempts
        self.ignore_ambiguous = ignore_ambiguous

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path, mask_path = self.samples[index]
        image = _read_image(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        raw = _read_image(label_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[:2] != raw.shape:
            raise ValueError(f"Image/label size mismatch: {image_path}")
        label = np.zeros(raw.shape, dtype=np.float32)
        threshold = self.label_threshold * 255.0
        if self.ignore_ambiguous:
            label[(raw > 0) & (raw < threshold)] = 2
        label[raw >= threshold] = 1
        if mask_path is not None:
            mask = _read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape != raw.shape:
                raise ValueError(f"Image/mask size mismatch: {mask_path}")
            label[mask < threshold] = 2
        if self.crop_size is not None:
            image, label = self._crop(image, label)
        if self.augment:
            for axis in (0, 1):
                if random.random() < 0.5:
                    image, label = np.flip(image, axis), np.flip(label, axis)
        if min(label.shape) < 9:
            raise ValueError("RCF needs image dimensions >= 9")
        image = (image - BGR_MEAN).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(
            np.ascontiguousarray(label[None]))

    def _crop(self, image, label):
        size = self.crop_size
        h, w = label.shape
        bottom, right = max(0, size - h), max(0, size - w)
        if bottom or right:
            image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_REPLICATE)
            label = cv2.copyMakeBorder(label, 0, bottom, 0, right,
                                       cv2.BORDER_CONSTANT, value=2)
        h, w = label.shape
        best, best_count = None, -1
        attempts = self.crop_attempts if self.crop_mode == "random" else 1
        for _ in range(attempts):
            top = random.randint(0, h-size) if self.crop_mode == "random" else (h-size)//2
            left = random.randint(0, w-size) if self.crop_mode == "random" else (w-size)//2
            cropped = (image[top:top+size, left:left+size], label[top:top+size, left:left+size])
            count = np.count_nonzero(cropped[1] == 1)
            if count > best_count:
                best, best_count = cropped, count
            if count >= self.min_edge_pixels:
                break
        return best
