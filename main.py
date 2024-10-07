import os
import utils
import models
import data_loader
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision


def modelini():
	path = os.path.dirname(os.path.realpath(__file__))
	model = models.RCF()
	model.cpu()
	utils.load_pretrained(model, path + "/models/RCFcheckpoint_epoch12.pth")
	return model


def _image_cv2nn(img):
	img = np.float32(img)
	img = data_loader.prepare_image_cv2(img)
	img = torch.unsqueeze(torch.from_numpy(img).cpu(), 0)
	return img


def _image_nn2cv(img):
	return torch.squeeze(img.detach()).cpu().numpy()


class ModelGPU:
	def __init__(self):
		path = os.path.dirname(os.path.realpath(__file__))
		self.model = models.RCF()
		self.model.cuda()
		utils.load_pretrained(self.model, path + "/models/RCFcheckpoint_epoch12.pth", device='cuda')

	def get_model_edges(self, img):
		img_nn = _image_cv2nn(img)
		results = self.model(img_nn)
		return _image_nn2cv(results[-1])
