import math
import numpy as np

class Cropper:
	def __init__(self, model, crop=512, pad=64, display=False):
		self.crop = int(crop)
		self.pad = int(pad)
		self.display = display
		self.model = model
		self.image = None
		self.output = None
		self.progress = None

		if self.crop <= 0:
			raise ValueError("crop must be positive")

		if self.pad < 0:
			raise ValueError("pad must be non-negative")

		if self.crop <= 2 * self.pad:
			raise ValueError("crop must be greater than 2 * pad")

	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.image[y:y + dy, x:x + dx]
		edges_small = self.model(img_small)
		self.output[y + ddy:y + dy - ddy, x + ddx:x + dx - ddx] = edges_small[ddy:dy - ddy, ddx:dx - ddx]
		if self.progress is not None:
			self.progress.update(1)

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

		total = i_max * j_max

		for i in range(i_max):
			for j in range(j_max):
				x = step_x * j + shift_x
				y = step_y * i + shift_y
				self.get_crop_edge(x, y, dx, dy, ddx, ddy)

	def _count_crops(self, dx, dy, ddx, ddy):
		step_x = dx - 2 * ddx
		step_y = dy - 2 * ddy

		i_max = (self.sh[0] - 2 * ddy) // step_y
		j_max = (self.sh[1] - 2 * ddx) // step_x

		corners = 4
		vertical_edges = 2 * i_max  # left + right
		horizontal_edges = 2 * j_max  # top + bottom
		center = i_max * j_max

		return corners + vertical_edges + horizontal_edges + center

	def get_cropped_output(self, dx, dy, ddx, ddy):
		self.top_left_edges(dx, dy, ddx, ddy)
		self.bottom_right_edges(dx, dy, ddx, ddy)
		self.bottom_left_edges(dx, dy, ddx, ddy)
		self.top_right_edges(dx, dy, ddx, ddy)
		self.top_edges(dx, dy, ddx, ddy)
		self.right_edges(dx, dy, ddx, ddy)
		self.bottom_edges(dx, dy, ddx, ddy)
		self.left_edges(dx, dy, ddx, ddy)
		self.center_edges(dx, dy, ddx, ddy)
		return self.output

	@staticmethod
	def _make_progress(total, enabled, desc="llambdakern"):
		if not enabled:
			return None

		try:
			from tqdm.auto import tqdm
		except ImportError:
			print("tqdm is not installed; progress display disabled")
			return None

		return tqdm(total=total, desc=desc)

	def __call__(self, image):
		self.image = image
		self.sh = image.shape
		self.output = np.zeros(image.shape[0:2], np.float32)

		ddx = self.pad if self.crop < self.sh[1] else 0
		ddy = self.pad if self.crop < self.sh[0] else 0

		dx = self.crop if self.crop < self.sh[1] else self.sh[1]
		dy = self.crop if self.crop < self.sh[0] else self.sh[0]

		total = self._count_crops(dx, dy, ddx, ddy)
		self.progress = self._make_progress(total, self.display)

		try:
			output = self.get_cropped_output(dx, dy, ddx, ddy)
		finally:
			if self.progress is not None:
				self.progress.close()
				self.progress = None

		return output

