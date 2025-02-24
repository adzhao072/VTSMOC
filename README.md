# Voronoi Tree Search Boosted Multi-Objective Bayesian Optimization with Constraints (VTSMOC)
VTSMOC is a sample-efficient and computationally lightweight multi-objective Bayesian optimization method tailored for high-dimensional input spaces. VTSMOC employs a Voronoi tree structure to recursively partition the design space, transforming it into a binary space partition (BSP) tree. It explores the Pareto front (PF) through iterative path selection, branch expansion, and reward back-propagation operations. To achieve high sample efficiency and maintain low computational cost, we propose the Expected Pareto Front Improvement (EPFI) and Probability of Pareto Front Improvement (PPFI) acquisition functions. These functions facilitate the efficient exploration of the PF along the radial direction of the PF surface. The EPFI/PPFI acquisition functions are expressed in closed form using joint Gaussian processes (GPs) and have a significantly lower computational cost than the popular Expected Hypervolume Improvement (EHVI) method.


Please cite this package as follows: 

```
@ARTICLE{VTSMOC-article,
  author={Zhao, Aidong and Lyu, Ruiyu and Zhao, Xuyang and Bi, Zhaori and Yang, Fan and Yan, Changhao and Zhou, Dian and Su, Yangfeng and Zeng, Xuan},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={VTSMOC: An Efficient Voronoi Tree Search Boosted Multiobjective Bayesian Optimization With Constraints for High-Dimensional Analog Circuit Synthesis}, 
  year={2025},
  volume={44},
  number={3},
  pages={818-831},
  doi={10.1109/TCAD.2024.3455932}}
```


## Dependencies
--------------

    - Python == 3.9.16
    - numpy == 1.24.2
    - scipy == 1.10.1
    - setuptools == 53.0.0
    - torch == 2.0.0
    - gpytorch == 1.10
    - botorch == 0.9.2





## Examples

import numpy as np

from VTSMOC import VTSMOC

from benchmarks.DTLZ import DTLZ2

# Dimension of input space
dims = 50
# Number of objectives
n_obj = 2
# Number of constraints
n_con = 0

dtlz2 = DTLZ2(n_var=dims,n_obj=n_obj)

# Referece point for HV calculation
ref_max = np.array([6.0]*n_obj)

# Input space
bound = np.array([np.zeros(dims),np.ones(dims)])

# Objective function
def DTLZ2(x):

    xs = bound[0] + (bound[1] - bound[0]) * np.atleast_2d(x)  #normalize to [0,1]^d
    
    y = dtlz2._evaluate_F(xs)
    
    return y

# Lower bound of input space
lb = np.zeros(dims)
# Upper bound of input space
ub = np.ones(dims)
# Number of iterations
iteration = 2000
# Number of initial points
ninits = 50

# Batch size
batch_size=5
# Acquisition function
acq_func = 'PPFI'
# Expansion factor
expansion_factor = 0.0

# Search with VTSMOC
optimizer = VTSMOC(

                   lb,              # the lower bound of each problem dimensions

                   ub,              # the upper bound of each problem dimensions

                   dims,          # the problem dimensions

                   DTLZ2,               # function object to be optimized

                   n_obj,

                   n_con,

                   ninits,      # the number of random samples used in initializations 

                   iteration,    # maximal iterations

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




