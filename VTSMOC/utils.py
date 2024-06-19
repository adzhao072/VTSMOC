# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:03:49 2022

@author: Zhao Aidong
"""

import numpy as np
import time
import os

def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = np.arange(n)
    centers = centers / float(n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(np.arange(n))]

    perturbation = np.random.rand(n, dims) 
    perturbation = perturbation / float(n)
    points += perturbation
    return points

def to_size(x, lb, ub):
    return lb + (ub-lb) * x


def fast_non_dominated_sort(fX):
    assert fX.ndim == 2, "make sure the multi-objective input..."
    num_samples, num_objs = fX.shape
    
    if num_samples < 2:
        return np.array([1])
    else:
    #assert num_samples >= 2, "make sure more than two points..."
    
        rank = np.zeros(num_samples)
        S_idx = []
        n = np.zeros(num_samples)
        PF_idx = []
        for j in range(num_samples):
            tmpS_idx = []
            for k in range(num_samples):
                if dominate(fX[j],fX[k]):
                    tmpS_idx.append(k)
                elif dominate(fX[k],fX[j]):
                    n[j] += 1
            if n[j] == 0:
                rank[j] = 1
                PF_idx.append(j)
            S_idx.append(np.array(tmpS_idx))
        
        i = 1
        while len(PF_idx) > 0:
            Q_idx = []
            for pf in PF_idx:
                for idx in S_idx[pf]:
                    n[idx] -= 1
                    if n[idx] == 0:
                        rank[idx] = i + 1
                        Q_idx.append(idx)
            i += 1
            PF_idx = Q_idx
    return rank


def dominate(y1,Y):
    return np.bitwise_and((y1 <= Y).all(axis=-1),(y1 < Y).any(axis=-1))
    


class problem_transition():
    
    def __init__(self, functs, lb, ub, n_objs = 2, n_cons = 0, n_eqcons = 0, epsilon = 1e-3, logfile = 'result.csv'):
        
        self.lb = lb
        self.ub = ub
        self.functs = functs
        self.dims = len(lb)
        self.n_objs = n_objs
        self.n_cons = n_cons
        self.n_eqcons = n_eqcons
        self.epsilon = epsilon
        self.logfile = logfile
        
        if os.path.exists(logfile):
            os.remove(logfile)
        return 
    
    def evaluate(self, x):
        xs = self.lb + np.atleast_2d(x)*(self.ub-self.lb)
        ys = self.functs(xs)
        objs = ys[:,:self.n_objs]
        
        if self.n_eqcons+self.n_cons > 0:
            cons = np.zeros((ys.shape[0],self.n_eqcons+self.n_cons))
            #print('cons = ',ys[:,self.n_objs:self.n_objs+self.n_cons])
            cons[:,:self.n_cons] = ys[:,self.n_objs:self.n_objs+self.n_cons]
            if self.n_eqcons > 0 :
                cons[:,self.n_cons:self.n_cons+self.n_eqcons] = np.abs(ys[:,self.n_objs+self.n_cons:self.n_objs+self.n_cons+self.n_eqcons])-self.epsilon
        else:
            cons = None    
            
        with open(self.logfile,'a+') as f:
            for s in ys:
                f.write(str(time.time())+',')
                for t in s:
                    f.write(str(t)+',')
                f.write('\n')
                
        return objs, cons
    

    
if __name__ == "__main__":
    ys = np.random.randn(10,3)
    print(fast_non_dominated_sort(ys))
    


