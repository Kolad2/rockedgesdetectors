import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision

from . import utils, models, data_loader


def _image_cv2cpunn(img):
	img = np.float32(img)
	img = data_loader.prepare_image_cv2(img)
	img = torch.unsqueeze(torch.from_numpy(img).cpu(), 0)
	return img


def _image_cv2gpunn(img):
	img = np.float32(img)
	img = data_loader.prepare_image_cv2(img)
	img = torch.unsqueeze(torch.from_numpy(img).cuda(), 0)
	return img


def _image_cpunn2cv(img):
	return torch.squeeze(img.detach()).cpu().numpy()


class ModelRCF:
	def __init__(self, path_to_model=None, device="cuda"):
		if path_to_model is None:
			path = os.path.dirname(os.path.realpath(__file__))
			path_to_model = path + "/models/RCFcheckpoint_epoch12.pth"
		self.model = models.RCF()
		self.model.cuda()
		utils.load_pretrained(self.model, path_to_model, device='cuda')

	def __call__(self, image):
		return self.get_model_edges(image)

	def get_model_edges(self, image):
		img_nn = _image_cv2gpunn(image)
		results = self.model(img_nn)
		return _image_cpunn2cv(results[-1])
