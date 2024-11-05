import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .pidinet.utils import cross_entropy_loss_RCF
from . import ModelPiDiNet
from .dataset import Dataset

import matplotlib.pyplot as plt


class ModelTrain:
	grad_com_size = 24

	@classmethod
	def train(
		cls,
		train_loader: DataLoader,
		model: ModelPiDiNet,
		optimizer: torch.optim.Optimizer
	):
		print("train start")
		model.model.train()
		optimizer.zero_grad()

		counter = 0

		num_iterations = len(train_loader)
		for i, (image, label) in enumerate(train_loader):
			print(i, "/", num_iterations)
			image = image.cuda(non_blocking=True)
			label = label.cuda(non_blocking=True)
			outputs = model(image)

			loss = 0
			if isinstance(outputs, list):
				for output in outputs:
					loss = loss + cross_entropy_loss_RCF(output, label, 1.1)

			counter = counter + 1
			loss = loss / cls.grad_com_size
			loss.backward()
			if counter == cls.grad_com_size:
				optimizer.step()
				optimizer.zero_grad()
				counter = 0
