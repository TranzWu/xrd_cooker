from xrd_cooker.xrd_analyzer import Analyzer
from xrd_cooker.parse import Parser
import taichi as ti
import numpy as np
import os
import matplotlib.pyplot as plt

os.system("mpirun -np 4 lmp -in init.in > out_init")
ti.init()

#os.system('mpirun -np 4 lmp -in init.in')
analyser = Analyzer('test.xrd')
parser = Parser('data_out')
N = parser.num_atoms


x = ti.Vector.field(3, float, N, needs_grad=True)
U = ti.field(float, (), needs_grad=True)
dx = 0.1

'''
define the 2theta to be 16.4, 17.8, 22.8, 26.1 respectively
This corresponds to the (011), (010), (110), (100) diffraction places
'''

#peaks = {16.4:0.1, 17.8:0.4, 22.8:0.2, 26.1:0.3}

def prob(x, m, s):
	return 1/((2 * np.pi * s * s)**0.5) * np.exp(-(x - m)**2/(2*s*s))

def cook(x):
	return 0.1 * prob(x, 14.4, 0.1) + 0.4 * prob(x, 17.8, 0.1) + 0.2 * prob(x, 22.8, 0.1) + 0.3 * prob(x, 26.1, 0.1)

cook_value = [cook(i) for i in analyser.angle]


def object_function():
	for i in range(analyser.number_of_bin):
		U[None] += (analyser.intensity[i] - cook_value[i])**2


#print(analyser.angle)

@ti.kernel
def advance():
	for i in x:
		x[i] += -x.grad[i]

@ti.kernel
def init():
	for i in range(N):
		x[i] = parser.coordinates[i]

def step():
	with ti.Tape(U):
		object_function()
	advance()

step()
xxx = []
for i in range(10):
	print(x.grad.to_numpy())
	analyser.plot(f'{i}')
	parser.update_coordinate(new=x.to_numpy())
	parser.write('data_in')

	os.system("mpirun -np 4 lmp -in cal_xrd.in > out_xrd")
	step()
	analyser = Analyzer('test.xrd')
	parser = Parser('data_out')

	xxx.append(U[None])
	#print(parser.coordinates[0])


print(xxx)

