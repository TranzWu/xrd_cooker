from xrd_cooker.xrd_analyzer import Analyzer
from xrd_cooker.parse import Parser
import numpy as np
import os
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import random

os.system("mpirun -np 4 lmp -in init.in > out_init")
os.system('mkdir plot')
parser = Parser('data_out')
analyser = Analyzer('test.xrd')

count = 0
loss_list = []

def black_box_function(*args, **kwags):     
    return analyser.calculate_loss() 

utility = UtilityFunction(kind="ucb", kappa=15.5, xi=0.0)

# Bounded region of parameter space
pbounds = {f'x_{i}_{j}':(0, 30) for i in range(parser.num_atoms) for j in range(3)}

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    random_state=1,
)

for i in range(100):
    updated_coordinate = optimizer.suggest(utility)
    parser.update_coordinate(updated_coordinate)
    parser.write('data_in')
    os.system("mpirun -np 4 lmp -in cal_xrd.in > out_xrd")

    analyser = Analyzer('test.xrd')
    parser = Parser('data_out')

    target = black_box_function(**updated_coordinate)
    optimizer.register(params=updated_coordinate, target=target)

    loss_list.append(target)

    analyser.plot(f'{i}')
    os.system(f'mv {i}.png plot/')
    count += 1

    print(target)

