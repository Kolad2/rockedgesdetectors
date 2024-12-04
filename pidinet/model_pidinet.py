import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from .models import PiDiNet
from .config import config_model

_normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)

_transform = transforms.Compose(
	[transforms.ToTensor(), _normalize]
)


def _ndarray_to_pidinet(ndarray_image):
	return _transform(ndarray_image).unsqueeze(0).cuda()


def _pidinet_to_ndarray(pidinet_edges):
	return torch.squeeze(pidinet_edges[-1]).detach().cpu().numpy()


class ModelPiDiNet:

	@staticmethod
	def output_to_ndarray(pidinet_edges):
		return torch.squeeze(pidinet_edges[-1]).detach().cpu().numpy()

	def __init__(self, checkpoint_path=None):
		pdcs = config_model('carv4')
		self.model = PiDiNet(60, pdcs, dil=24, sa=True)
		self.model = torch.nn.DataParallel(self.model).cuda()
		if checkpoint_path is not None:
			checkpoint = torch.load(checkpoint_path, map_location='cuda')
			self.model.load_state_dict(checkpoint['state_dict'])

	def get_weights(self):
		return PiDiNet.get_weights(self.model)

	def __call__(self, image):
		if isinstance(image, np.ndarray):
			return self.get_ndarray_model_edges(image)
		if isinstance(image, torch.Tensor):
			return self.model(image)

	def get_ndarray_model_edges(self, image):
		result_pidinet = self.model(_ndarray_to_pidinet(image))
		return _pidinet_to_ndarray(result_pidinet)
