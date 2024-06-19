# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:55:28 2023

@author: ZhaoAidong
"""

import numpy as np 
from .utils import fast_non_dominated_sort
import torch
import random
from botorch.utils.multi_objective.hypervolume import Hypervolume



def cluster_fx(fs,Cons,batch_size,PF_shape,num_clusters = None,device='cpu'):
    num_pnts, n_objs = fs.shape
    if num_clusters is None:
        num_clusters = num_pnts//3
    num_clusters = min(max(2*batch_size,num_clusters),num_pnts)
    clusters_cnt = num_pnts
    
    if Cons is None:
        fgx = fs
    else:
        fgx = np.hstack([fs,Cons])
    
    distance = np.inf * np.ones((num_pnts,num_pnts))
    #assert (fs>=0).all()
    for j in range(num_pnts):
        for i in range(j):
            distance[j,i] = distance[i,j] = np.linalg.norm(fgx[i]-fgx[j])
    
    
    idx_list = [[i] for i in range(num_pnts)]
    while clusters_cnt > num_clusters:
        min_dist = np.inf
        
        for j in range(len(idx_list)):
            for k in range(j+1,len(idx_list)):
                dist_jk = distance[idx_list[j],:][:,idx_list[k]]
                d = dist_jk.max()
                #print('d= ',d)
                if d < min_dist:
                    combain_pair = [j,k]
                    min_dist = d
                     
        new_node = idx_list[combain_pair[0]] + idx_list[combain_pair[1]]
        idx_list = [s for i,s in enumerate(idx_list) if not i in combain_pair]
        idx_list.append(new_node)
        clusters_cnt -= 1
    num_clusters = clusters_cnt 

    hv_computer = Hypervolume(ref_point = -torch.ones(n_objs,device=device))
    Ys = -torch.tensor(fs,device=device)
    total_hv = hv_computer.compute(Ys)
    PF_indicator = torch.ones(len(fs),device=device) > 0
    
    ###tournament selection #####
    
    N_pop = min(max(2*num_clusters//3,4*batch_size//3),num_clusters)
    
    hvc_max = -np.inf * np.ones(num_clusters) 
    #G_min = np.inf * np.ones(num_clusters)
    #q_value = -np.inf * np.ones(num_clusters)
    best_idx = -1 * np.ones(num_clusters,dtype=int)
    random_selection = random.sample(range(num_clusters), N_pop)
    for j in range(N_pop):
        
        idx = random_selection[j]
        hvc_m_j = -np.inf
        #q_m_j = -np.inf
        best_k = -1
        #print('idxlist=',idx_list[idx])
        for k in idx_list[idx]:
            
            if Cons is None:
            
                PF_subset = PF_indicator.clone().detach()
                PF_subset[k] = False
                hvc = total_hv - hv_computer.compute(Ys[PF_subset])
                
            else:
                if (Cons[k]>0).any():
                    hvc = -np.sum(np.maximum(Cons[k],0.0))
                else:
                    PF_subset = PF_indicator.clone().detach()
                    PF_subset[k] = False
                    hvc = total_hv - hv_computer.compute(Ys[PF_subset])
                    
            if hvc > hvc_m_j:
                hvc_m_j = hvc
                best_k = k
                
        hvc_max[j] = hvc_m_j
        #q_value[j] = q_m_j
        best_idx[j] = best_k
        
    
    sorted_groups = np.argsort(-hvc_max)
    print('Q value = ', hvc_max[sorted_groups[:batch_size]])
    
    
    select_flag = np.zeros(num_pnts) > 1
    select_flag[best_idx[sorted_groups[:batch_size]]] = True
    
    
    return select_flag







def PF_surface_estimination(fx):
    N_objectives = fx.shape[1]
    PF_pnts = fx.shape[0]
    if PF_pnts <= N_objectives:
        return 'linear'
    else:
        z_star = fx.min(axis = 0)
        z_nad = fx.max(axis = 0)
        v_std = 1. / np.sqrt(N_objectives) * np.ones(N_objectives)
        z = (fx-z_star) / (z_nad - z_star)
        
        dev = np.array([ np.linalg.norm(s - v_std) for s in z])
        sort_idx = np.argsort(dev)
        vectors = z[sort_idx[:N_objectives]]
        q_value = np.linalg.norm(vectors.sum(axis=0))/N_objectives * 2/ np.sqrt(N_objectives)
        if q_value < 0.9:
            return 'convex'
        elif q_value >1.1:
            return 'concave'
        else:
            return 'linear'
    






if __name__ == "__main__":
    a=np.random.randn(18,3)
    a[:,1] *= 3
    print(a)
    labels = fast_non_dominated_sort(a)
    print(labels)

   
    
    #print()
