# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:27:29 2023

@author: ZhaoAidong
"""
import numpy as np
import time
import os



from VTSMOC import VTSMOC
from benchmarks.DTLZ import DTLZ2,DTLZ3,DTLZ4,DTLZ5,DTLZ6


dims = 50
n_obj = 2
dtlz2 = DTLZ2(n_var=dims,n_obj=n_obj)


#dims = 60


ref_max = np.array([6.0]*n_obj)

bound = np.array([np.zeros(dims),np.ones(dims)])


def DTLZ2(x):
    xs = bound[0] + (bound[1] - bound[0]) * np.atleast_2d(x)  #normalize to [0,1]^d
    y = dtlz2._evaluate_F(xs)
    print('y = ',y)
    '''
    with open('result.csv','a+') as f:
        for s in y:
            f.write(str(time.time())+',')
            for t in s:
                f.write(str(t)+',')
            f.write('\n')
    '''
    return y




if os.path.exists('result.csv'):
    os.remove('result.csv')
    


#dims = 200
lb = np.zeros(dims)
ub = np.ones(dims)
iteration = 100

#n_init = 50

f = DTLZ2
ninits = 50
n_objs = n_obj
n_cons = 0
#n_eqcons = 0
batch_size=5
acq_func = 'PPFI'
expansion_factor = 0.0


optimizer = VTSMOC(
                   lb,              # the lower bound of each problem dimensions
                   ub,              # the upper bound of each problem dimensions
                   dims,          # the problem dimensions
                   f,               # function object to be optimized
                   n_objs,
                   n_cons,
                   #n_eqcons,
                   ninits,      # the number of random samples used in initializations 
                   iteration,    # maximal iteration to evaluate f(x)
                   batch_size,
                   ref_max,
                   leaf_size = 200,  # tree leaf size
                   kernel_type = 'Matern',   # kernel for GP model
                   acq_func = acq_func,
                   expansion_factor = expansion_factor,
                   use_cuda = False,     #train GP on GPU
                   set_greedy = False    # greedy setting
                   )

optimizer.search()

#print(RoverTrajectory(0.9*np.random.randn(dim)))