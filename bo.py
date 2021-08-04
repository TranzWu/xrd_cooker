from xrd_cooker.xrd_analyzer import Analyzer
from xrd_cooker.parse import Parser
import numpy as np
import os
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import random

os.system("mpirun -np 4 lmp -in init.in > out_init")
os.system('mkdir plot')
parser = Parser('data_out')
analyser = Analyzer('test.xrd')

def black_box_function(*args, **kwags):   
    return analyser.calculate_loss()


# Bounded region of parameter space
pbounds = {f'x_{i}_{j}':(0, 30) for i in range(parser.num_atoms) for j in range(3)}
#print(type(pbounds))
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

loss_list = []
for i in range(100):
    print('asdf\n\n\n')

    optimizer.maximize(
        init_points=10,
        n_iter=1,
    )

    updated_coordinate = optimizer.res[-1]['params']
    parser.update_coordinate(updated_coordinate)
    parser.write('data_in')

    os.system("mpirun -np 4 lmp -in cal_xrd.in > out_xrd")
    analyser = Analyzer('test.xrd')
    parser = Parser('data_out')

    os.system(f'mv {i}.png plot/')
    if i%10 == 0:
        analyser.plot(f'{i}')
        os.system(f'mv {i}.png plot/')
    loss_list.append(optimizer.res[-1]['target'])

print(loss_list)
# updated_coordinate = optimizer.res[-1]['params']
# print(updated_coordinate)
