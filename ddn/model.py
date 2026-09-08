"""Pure-PyTorch inference implementation of DDN-M36 for edge detection."""

from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import DDNCAFormerM36


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.attention1 = nn.Identity()
        self.conv2 = Conv2dReLU(out_channels, out_channels)
        self.attention2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat((x, skip), dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.attention2(self.conv2(self.attention1(self.conv1(x))))


class UnetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Reversed feature channels: 384, 192, 96, 32, 16.
        self.conv41 = DecoderBlock(384, 192, 128)
        self.conv31 = DecoderBlock(192, 96, 64)
        self.conv32 = DecoderBlock(128, 64, 64)
        self.conv21 = DecoderBlock(96, 32, 32)
        self.conv22 = DecoderBlock(64, 32, 32)
        self.conv23 = DecoderBlock(64, 32, 32)
        self.conv11 = DecoderBlock(32, 16, 16)
        self.conv12 = DecoderBlock(32, 16, 16)
        self.conv13 = DecoderBlock(32, 16, 16)
        self.conv14 = DecoderBlock(32, 16, 16)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        features = list(features)
        features[0] = self.conv11(features[1], features[0])
        features[1] = self.conv21(features[2], features[1])
        features[2] = self.conv31(features[3], features[2])
        features[3] = self.conv41(features[4], features[3])

        features[0] = self.conv12(features[1], features[0])
        features[1] = self.conv22(features[2], features[1])
        features[2] = self.conv32(features[3], features[2])

        features[0] = self.conv13(features[1], features[0])
        features[1] = self.conv23(features[2], features[1])
        return self.conv14(features[1], features[0])


class DDN(nn.Module):
    """DDN-M36 loaded from the authors' released BSDS500 checkpoint.

    The module returns the mean and standard deviation edge logits. Use
    :class:`NumpyDDNAdapter` for the final edge probability map.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        trainable: bool = False,
    ):
        super().__init__()
        with torch.device("meta"):
            self.encoder = DDNCAFormerM36(local_channels=16)
            self.decoder = UnetDecoder()
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(16, 1, 3, padding=1),
            )
            self.decoder_1 = UnetDecoder()
            self.segmentation_head_1 = nn.Sequential(
                nn.Conv2d(16, 1, 3, padding=1),
            )

        self._load_checkpoint(Path(checkpoint_path))
        self.requires_grad_(trainable)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DDN checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("The DDN checkpoint has no state_dict")
        self.load_state_dict(state, strict=True, assign=True)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        height, width = image.shape[-2:]
        features = self.encoder(image)

        mean = self.segmentation_head(self.decoder(*features))
        mean = F.interpolate(mean, (height, width), mode="bilinear")

        std = self.segmentation_head_1(self.decoder_1(*features))
        std = F.interpolate(std, (height, width), mode="bilinear")
        return mean, F.softplus(std)
