"""Inference-only PyTorch implementation of the released DiffusionEdge BSDS model."""

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .autoencoder import AutoencoderKL
from .unet import Unet


class _Config(dict):
    """Small attribute-access dictionary replacing the training-time config package."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _create_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(
        ddconfig={
            "double_z": True,
            "z_channels": 3,
            "resolution": [320, 320],
            "in_channels": 1,
            "out_ch": 1,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        lossconfig={},
        embed_dim=3,
    )


def _create_unet() -> Unet:
    config = _Config(
        cond_net="swin",
        cond_pe=False,
        cond_feature_size=[80, 80],
        fix_bb=True,
        input_size=[80, 80],
        num_pos_feats=128,
    )
    return Unet(
        dim=128,
        channels=3,
        out_mul=1,
        dim_mults=[1, 2, 4, 4],
        cond_in_dim=3,
        cond_dim=128,
        cond_dim_mults=[2, 4],
        window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
        window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
        fourier_scale=16,
        cfg=config,
    )


class DiffusionEdgeBSDS(nn.Module):
    """The authors' standard PyTorch BSDS model, reduced to inference code."""

    input_size = 320
    latent_channels = 3
    latent_down_ratio = 4

    def __init__(
        self,
        checkpoint_path: str | Path,
        sampling_timesteps: int = 1,
        denoise_last_step: bool = True,
    ):
        super().__init__()
        if sampling_timesteps <= 0:
            raise ValueError("sampling_timesteps must be positive")

        # Meta construction avoids allocating a second 1.2 GB initialized copy
        # before assigning the tensors from the released checkpoint.
        with torch.device("meta"):
            self.model = _create_unet()
            self.first_stage_model = _create_autoencoder()
            self.register_buffer("scale_factor", torch.tensor(0.3))

        self.sampling_timesteps = int(sampling_timesteps)
        self.denoise_last_step = bool(denoise_last_step)
        self.eps = 1e-4

        self._load_checkpoint(Path(checkpoint_path))
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DiffusionEdge checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, Mapping):
            raise ValueError("The DiffusionEdge checkpoint has no EMA state")

        prefix = "ema_model."
        released_state = {
            name[len(prefix):]: tensor
            for name, tensor in ema_state.items()
            if name.startswith(prefix)
        }
        expected_names = set(self.state_dict())
        state = {
            name: tensor
            for name, tensor in released_state.items()
            if name in expected_names
        }
        missing = expected_names.difference(state)
        if missing:
            missing_text = ", ".join(sorted(missing)[:10])
            raise ValueError(
                "The DiffusionEdge checkpoint is missing model tensors: "
                f"{missing_text}"
            )

        self.load_state_dict(state, strict=True, assign=True)

    @staticmethod
    def _pred_x0_from_xt(
        current: torch.Tensor,
        noise: torch.Tensor,
        coefficient: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time = time.reshape(coefficient.shape[0], 1, 1, 1)
        return current - coefficient * time - torch.sqrt(time) * noise

    @staticmethod
    def _pred_previous_from_current(
        current: torch.Tensor,
        noise: torch.Tensor,
        coefficient: torch.Tensor,
        time: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        time = time.reshape(coefficient.shape[0], 1, 1, 1)
        step = step.reshape(coefficient.shape[0], 1, 1, 1)
        mean = current - coefficient * step - step / torch.sqrt(time) * noise
        sigma = torch.sqrt(step * (time - step) / time)
        return mean + sigma * torch.randn_like(mean)

    def _time_steps(self, device: torch.device) -> torch.Tensor:
        step = 1.0 / self.sampling_timesteps
        steps = torch.full(
            (self.sampling_timesteps,),
            step,
            device=device,
        )
        if self.denoise_last_step:
            steps = torch.cat((
                steps[:-1],
                torch.tensor([step - self.eps, self.eps], device=device),
            ))
        return steps

    def _sample_latent(self, condition: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = condition.shape
        shape = (
            batch,
            self.latent_channels,
            height // self.latent_down_ratio,
            width // self.latent_down_ratio,
        )
        current = torch.randn(shape, device=condition.device, dtype=condition.dtype)
        current_time = torch.ones((batch,), device=condition.device, dtype=condition.dtype)
        time_steps = self._time_steps(condition.device).to(condition.dtype)

        for index, step_value in enumerate(time_steps):
            step = torch.full_like(current_time, step_value)
            if index == len(time_steps) - 1:
                step = current_time

            coefficient, noise = self.model(current, current_time, condition)
            x0 = self._pred_x0_from_xt(
                current,
                noise,
                coefficient,
                current_time,
            )
            coefficient = -x0
            current = self._pred_previous_from_current(
                current,
                noise,
                coefficient,
                current_time,
                step,
            )
            current_time = current_time - step

        return current

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        if image.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                "DiffusionEdge BSDS requires 320x320 input crops, got "
                f"{tuple(image.shape[-2:])}"
            )

        condition = image * 2.0 - 1.0
        latent = self._sample_latent(condition)
        latent = latent / self.scale_factor
        decoded = self.first_stage_model.decode(latent)
        return ((decoded + 1.0) * 0.5).clamp_(0.0, 1.0)
