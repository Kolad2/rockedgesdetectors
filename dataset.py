import cv2
from pathlib import Path

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms

_normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)

_transform = transforms.Compose(
	[transforms.ToTensor(), _normalize]
)



def _ndarray_to_pidinet(ndarray_image):
	return _transform(ndarray_image).unsqueeze(0).cuda()


class Dataset(data.Dataset):
	def __init__(self, path_lst: Path, path_root: Path = None):
		self.path_root = path_root if path_root is not None else path_lst.parent
		self.path_lst = str(path_lst)
		self.inputs_paths = []
		self.outputs_paths = []
		with open(self.path_lst, 'r') as file:
			self.len = 0
			for line in file:
				self.len = self.len + 1
				input_path, output_path = line.split()
				self.inputs_paths.append(Path(input_path))
				self.outputs_paths.append(Path(output_path))

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		input_image = cv2.imread(str(self.path_root / self.inputs_paths[index]))
		input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
		label = cv2.imread(str(self.path_root / self.outputs_paths[index]))
		label = label.astype(np.float32)
		if label.ndim == 3:
			label = np.squeeze(label[:, :, 0])
		assert label.ndim == 2
		threshold = 0.5
		label = label[np.newaxis, :, :]
		label[label == 0] = 0
		label[np.logical_and(label > 0, label < threshold)] = 2
		label[label >= threshold] = 1
		input_image = _transform(input_image)
		return input_image, label
