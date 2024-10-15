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
	def __init__(self, checkpoint_path):
		pdcs = config_model('carv4')
		self.model = PiDiNet(60, pdcs, dil=24, sa=True)
		self.model = torch.nn.DataParallel(self.model).cuda()

		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		self.model.load_state_dict(checkpoint['state_dict'])

	def __call__(self, image):
		return self.get_ndarray_model_edges(image)

	def get_ndarray_model_edges(self, image):
		result_pidinet = self.model(_ndarray_to_pidinet(image))
		return _pidinet_to_ndarray(result_pidinet)