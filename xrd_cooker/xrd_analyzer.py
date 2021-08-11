import matplotlib.pyplot as plt
import numpy as np


class Analyzer():
	def __init__(self, filename):
		self.filename = filename

	@property
	def input(self):
		with open(self.filename, 'r') as f:
			rad = f.readlines()
		rad = [l.split() for l in rad]
		return rad

	@property
	def header(self):
		return self.input[3]

	@property
	def number_of_bin(self):
		return int(self.header[1])

	@property
	def min_angle(self):
		return float(self.header[-2])

	@property
	def max_angle(self):
		return float(self.header[-1])

	@property
	def xrd_data(self):
		dt = self.input[4:4+self.number_of_bin]
		return dt

	@property
	def angle(self):
		return [float(l[1]) for l in self.xrd_data]

	@property
	def intensity(self):
		return [float(l[-1]) for l in self.xrd_data]

	def prob(self, x, m, s):
		return 1/((2 * np.pi * s * s)**0.5) * np.exp(-(x - m)**2/(2*s*s))

	def cook(self, x):
		return 0.1 * self.prob(x, 14.4, 0.1) + \
		0.4 * self.prob(x, 17.8, 0.1) + \
		0.2 * self.prob(x, 22.8, 0.1) + \
		0.3 * self.prob(x, 26.1, 0.1)

	@property
	def recepie(self):
		recepie = np.array([self.cook(i) for i in self.angle])
		return recepie/np.linalg.norm(recepie)

	def calculate_loss(self):
		loss = 0
		for i in range(self.number_of_bin):
			loss += (self.recepie[i] - self.intensity[i])**2
		return -loss

	def plot(self, out, recepie=False):
		plt.clf()
		plt.plot(self.angle, self.intensity, label='lammps')
		if recepie:
			plt.plot(self.angle, self.recepie, label='recepie')
		plt.legend()
		plt.savefig(out)

		