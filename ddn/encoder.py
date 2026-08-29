"""Inference subset of the CAFormer-M36 encoder used by DDN.

Adapted from Li-yachuan/DDN and MetaFormer Baselines, distributed under the
Apache License 2.0. Only the layers present in the released DDN-M36 checkpoint
are retained, so this module has no dependency on timm.
"""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pre_norm=None,
        post_norm=None,
        pre_permute: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return self.post_norm(x)


class Scale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class StarReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x).square() + self.bias


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        num_heads: int | None = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads or max(1, dim // head_dim)
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _channels = x.shape
        tokens = height * width
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(
            batch,
            height,
            width,
            self.attention_dim,
        )
        return self.proj_drop(self.proj(x))


class LayerNormGeneral(nn.Module):
    def __init__(
        self,
        affine_shape: int,
        normalized_dim=(-1,),
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.weight = nn.Parameter(torch.ones(affine_shape))
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(self.normalized_dim, keepdim=True)
        variance = centered.square().mean(self.normalized_dim, keepdim=True)
        x = centered / torch.sqrt(variance + self.eps)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=None,
            eps=self.eps,
        )


class SepConv(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 2,
        kernel_size: int = 7,
        padding: int = 3,
        bias: bool = False,
        **_kwargs,
    ):
        super().__init__()
        hidden_channels = expansion_ratio * dim
        self.pwconv1 = nn.Linear(dim, hidden_channels, bias=bias)
        self.act1 = StarReLU()
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
            bias=bias,
        )
        self.act2 = nn.Identity()
        self.pwconv2 = nn.Linear(hidden_channels, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.pwconv1(x))
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.pwconv2(self.act2(x))


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        drop: float = 0.0,
        bias: bool = False,
        **_kwargs,
    ):
        super().__init__()
        hidden_features = mlp_ratio * dim
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = StarReLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        token_mixer,
        res_scale_init_value: float | None,
    ):
        super().__init__()
        self.norm1 = LayerNormWithoutBias(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.drop_path1 = nn.Identity()
        self.layer_scale1 = nn.Identity()
        self.res_scale1 = (
            Scale(dim, res_scale_init_value)
            if res_scale_init_value is not None
            else nn.Identity()
        )
        self.norm2 = LayerNormWithoutBias(dim)
        self.mlp = Mlp(dim=dim)
        self.drop_path2 = nn.Identity()
        self.layer_scale2 = nn.Identity()
        self.res_scale2 = (
            Scale(dim, res_scale_init_value)
            if res_scale_init_value is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_scale1(x) + self.token_mixer(self.norm1(x))
        return self.res_scale2(x) + self.mlp(self.norm2(x))


class DDNCAFormerM36(nn.Module):
    """Three-stage CAFormer-M36 feature encoder from the released DDN model."""

    def __init__(self, local_channels: int = 16):
        super().__init__()
        depths = (3, 12, 18)
        dims = (96, 192, 384)
        token_mixers = (SepConv, SepConv, Attention)
        residual_scales = (None, None, 1.0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, local_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(local_channels, local_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = [local_channels, local_channels * 2, *dims]

        norm = partial(LayerNormGeneral, bias=False, eps=1e-6)
        self.downsample_layers = nn.ModuleList((
            Downsampling(3, 96, 7, stride=4, padding=2, post_norm=norm),
            Downsampling(
                96,
                192,
                3,
                stride=2,
                padding=1,
                pre_norm=norm,
                pre_permute=True,
            ),
            Downsampling(
                192,
                384,
                3,
                stride=2,
                padding=1,
                pre_norm=norm,
                pre_permute=True,
            ),
        ))
        self.stages = nn.ModuleList(
            nn.Sequential(*(
                MetaFormerBlock(dim, mixer, scale)
                for _ in range(depth)
            ))
            for depth, dim, mixer, scale in zip(
                depths,
                dims,
                token_mixers,
                residual_scales,
            )
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.conv1(x)]
        features.append(self.conv2(features[0]))
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x.permute(0, 3, 1, 2))
        return features
