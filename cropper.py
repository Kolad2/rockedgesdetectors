import math
import numpy as np


class Cropper:
	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.img[y:y + dy, x:x + dx]
		self.edges[y + ddy:y + dy - ddy, x + ddx:x + dx - ddx] = self.model.get_model_edges(img_small)[ddy:dy - ddy, ddx:dx - ddx]

	def get_full_edge(self, dx, dy, ddx, ddy):
		i_max = math.floor((self.sh[0] - 2 * ddy) / (dy - 2 * ddy))
		j_max = math.floor((self.sh[1] - 2 * ddx) / (dx - 2 * ddx))
		for i in range(0, i_max):
			for j in range(0, j_max):
				if (i*i_max + j) % 1 == 0:
					print(i*i_max + j,"/",j_max*i_max)
				self.get_crop_edge((dx - 2 * ddx) * i, (dy - 2 * ddy) * j, dx, dy, ddx, ddy)
			y = (dy - 2 * ddy) * i_max
			self.get_crop_edge((dx - 2 * ddx) * i, y, dx, self.sh[0] - y, ddx, ddy)

		return self.edges
		# for j in range(0, i_max):
		# 	x = (dx - 2 * ddx) * j_max
		# 	self.get_crop_edge(x, (dy - 2 * ddy) * j, self.sh[1] - x, dy, ddx, ddy)

	def __init__(self, model, img):
		self.img = img
		self.sh = img.shape
		self.model = model
		self.edges = np.zeros(img.shape[0:2], np.float32)