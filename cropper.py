import math
import numpy as np


class Cropper:
	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.img[y:y + dy, x:x + dx]
		self.edges[y + ddy:y + dy - ddy, x + ddx:x + dx - ddx] = self.model(img_small)[ddy:dy - ddy, ddx:dx - ddx]

	def get_cropped_edges(self, dx, dy, ddx, ddy):
		i_max = math.floor((self.sh[0] - 2 * ddy) / (dy - 2 * ddy))
		j_max = math.floor((self.sh[1] - 2 * ddx) / (dx - 2 * ddx))
		for i in range(0, i_max):
			for j in range(0, j_max):
				if (i*i_max + j + 1) % 1 == 0:
					print(i*i_max + j + 1,"/",j_max*i_max)
				x = (dx - 2 * ddx) * j
				y = (dy - 2 * ddy) * i
				self.get_crop_edge(x, y, dx, dy, ddx, ddy)

		_x = (dx - 2 * ddx) * j_max
		_y = (dy - 2 * ddy) * i_max
		_dx = self.sh[1] - _x
		_dy = self.sh[0] - _y
		for j in range(0, j_max):
			x = (dx - 2 * ddx) * j
			self.get_crop_edge(x, _y, dx, _dy, ddx, ddy)

		for i in range(0, i_max):
			y = (dy - 2 * ddy) * i
			self.get_crop_edge(_x, y, _dx, dy, ddx, ddy)
		self.get_crop_edge(_x, _y, _dx, _dy, ddx, ddy)
		return self.edges

	def __init__(self, model, img=None, crop=512, pad=64):
		self.img = img
		self.sh = img.shape if img is not None else None
		self.model = model
		self.edges = np.zeros(img.shape[0:2], np.float32) if img is not None else None
		self.crop = crop
		self.pad = pad

	def __call__(self, image):
		self.img = image
		self.sh = image.shape
		self.edges = np.zeros(image.shape[0:2], np.float32)
		return self.get_cropped_edges(self.crop, self.crop, self.pad, self.pad)