import math
import numpy as np

from scripts.edge_detect import result


class Cropper:
	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.img[y:y + dy, x:x + dx]
		edges_small = self.model(img_small)
		self.edges[y + ddy:y + dy - ddy, x + ddx:x + dx - ddx] = edges_small[ddy:dy - ddy, ddx:dx - ddx]

	def bottom_right_edges(self, dx, dy, ddx, ddy):
		x, y = self.sh[1] - dx, self.sh[0] - dy
		self.get_crop_edge(x, y, dx, dy, 0, 0)

	def top_left_edges(self, dx, dy, ddx, ddy):
		x, y = 0, 0
		self.get_crop_edge(x, y, dx, dy, 0, 0)
	
	def bottom_left_edges(self, dx, dy, ddx, ddy):
		x, y = 0, self.sh[0] - dy
		self.get_crop_edge(x, y, dx, dy, 0, 0)

	def top_right_edges(self, dx, dy, ddx, ddy):
		x, y = self.sh[1] - dx, 0
		self.get_crop_edge(x, y, dx, dy, 0, 0)

	def right_edges(self, dx, dy, ddx, ddy):
		step_y = dy - 2 * ddy
		i_max = (self.sh[0] - 2 * ddy) // step_y
		shift_y = (self.sh[0] - i_max * step_y) // 2 - ddy
		x = self.sh[1] - dx
		for i in range(0, i_max):
			y = step_y * i + shift_y
			self.get_crop_edge(x, y, dx, dy, 0, ddy)

	def left_edges(self, dx, dy, ddx, ddy):
		step_y = dy - 2 * ddy
		i_max = (self.sh[0] - 2 * ddy) // step_y
		shift_y = (self.sh[0] - i_max * step_y) // 2 - ddy
		x = 0
		for i in range(0, i_max):
			y = step_y * i + shift_y
			self.get_crop_edge(x, y, dx, dy, 0, ddy)

	def bottom_edges(self, dx, dy, ddx, ddy):
		step_x = dx - 2 * ddx
		j_max = (self.sh[1] - 2 * ddx) // step_x
		shift_x = (self.sh[1] - j_max * step_x) // 2 - ddx
		y = self.sh[0] - dy
		for j in range(0, j_max):
			x = step_x * j + shift_x
			self.get_crop_edge(x, y, dx, dy, ddx, 0)

	def top_edges(self, dx, dy, ddx, ddy):
		step_x = dx - 2 * ddx
		j_max = (self.sh[1] - 2 * ddx) // step_x
		shift_x = (self.sh[1] - j_max*step_x) // 2 - ddx
		x, y = shift_x, 0
		for j in range(0, j_max):
			self.get_crop_edge(x, y, dx, dy, ddx, 0)
			x += step_x
	
	def center_edges(self, dx, dy, ddx, ddy):
		step_x = dx - 2 * ddx
		step_y = dy - 2 * ddy
		i_max = (self.sh[0] - 2 * ddy) // step_y
		j_max = (self.sh[1] - 2 * ddx) // step_x
		shift_x = (self.sh[1] - j_max * step_x) // 2 - ddx
		shift_y = (self.sh[0] - i_max * step_y) // 2 - ddy
		for i in range(0, i_max):
			for j in range(0, j_max):
				if (i * i_max + j + 1) % 1 == 0:
					print(i * i_max + j + 1, "/", j_max * i_max)
				x = step_x * j + shift_x
				y = step_y * i + shift_y
				self.get_crop_edge(x, y, dx, dy, ddx, ddy)

	def get_cropped_edges(self, dx, dy, ddx, ddy):
		self.top_left_edges(dx, dy, ddx, ddy)
		self.bottom_right_edges(dx, dy, ddx, ddy)
		self.bottom_left_edges(dx, dy, ddx, ddy)
		self.top_right_edges(dx, dy, ddx, ddy)
		self.top_edges(dx, dy, ddx, ddy)
		self.right_edges(dx, dy, ddx, ddy)
		self.bottom_edges(dx, dy, ddx, ddy)
		self.left_edges(dx, dy, ddx, ddy)
		self.center_edges(dx, dy, ddx, ddy)
		return self.edges

	def __init__(self, model, crop=512, pad=64):
		self.model = model
		self.crop = crop
		self.pad = pad

	def __call__(self, image):
		self.img = image
		self.sh = image.shape
		self.edges = np.zeros(image.shape[0:2], np.float32)
		ddx = self.pad
		ddy = self.pad
		dx = self.crop if self.crop >= self.sh[1] else self.sh[1] and ddx = 0
		dy = self.crop if self.crop >= self.sh[0] else self.sh[0] and ddy = 0
		return self.get_cropped_edges(dx, dy, ddx, ddy)