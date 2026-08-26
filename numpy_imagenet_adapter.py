from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch import nn
from torchvision import transforms


ModelOutput = Any
OutputSelector = Callable[[ModelOutput], torch.Tensor]


def _identity(output: ModelOutput) -> torch.Tensor:
	return output


class NumpyImagenetAdapter(nn.Module):
	"""Adapt NumPy images to an ImageNet-pretrained PyTorch module."""

	def __init__(
		self,
		module: nn.Module,
		output_selector: Optional[OutputSelector] = None,
	):
		super().__init__()
		self.module = module
		self.output_selector = output_selector or _identity
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		])

	@property
	def device(self) -> torch.device:
		parameter = next(self.module.parameters(), None)
		if parameter is not None:
			return parameter.device

		buffer = next(self.module.buffers(), None)
		if buffer is not None:
			return buffer.device

		return torch.device("cpu")

	def forward(
		self,
		image: Union[np.ndarray, torch.Tensor],
	) -> Union[np.ndarray, ModelOutput]:
		if isinstance(image, torch.Tensor):
			return self.module(image)

		if not isinstance(image, np.ndarray):
			raise TypeError(
				"image must be a numpy.ndarray or torch.Tensor, "
				f"got {type(image).__name__}"
			)

		input_tensor = self.transform(image).unsqueeze(0).to(self.device)
		output = self.output_selector(self.module(input_tensor))
		if not isinstance(output, torch.Tensor):
			raise TypeError(
				"output_selector must return a torch.Tensor, "
				f"got {type(output).__name__}"
			)

		return torch.squeeze(output).detach().cpu().numpy()
