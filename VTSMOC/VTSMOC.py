# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 15:04:49 2022

@author: Zhao Aidong
"""

import numpy as np

from .TreeNode import TreeNode
from .utils import latin_hypercube, to_size, problem_transition, fast_non_dominated_sort
from botorch.utils.multi_objective.hypervolume import Hypervolume
from .LBFGS_torch import acq_min_msp
from .approximate_PF import PF_surface_estimination
import torch


class VTSMOC(object):
    #############################################

    def __init__(self, lb, ub, dims, func, n_objs, n_cons, ninits, iteration, batch_size=1, ref_point=None, epsilon = None, leaf_size = 20, kernel_type = "rbf", acq_func = 'PPFI', expansion_factor = 0.0, use_cuda = False, set_greedy=False, logfile = 'result.csv',use_ard = False):
        assert ninits <= leaf_size
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        
        self.sigma_x                 =  0.000001
        self.batch_size              =  batch_size
        
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        
        self.curt_best_value         =  float("inf")
        #self.curt_best_sample        =  None
        self.sample_counter          =  0
        self.iterations              =  iteration
        
        self.LEAF_SIZE               =  leaf_size
        self.kernel_type             =  kernel_type
        self.use_ard                 =  use_ard
        
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        
        self.n_objs = n_objs
        self.n_cons = n_cons
        
        self.acq_func = acq_func
        self.expansion_factor = expansion_factor
        
        #self.n_cons = n_cons + n_eqcons
        self.epsilon = 1e-6 if epsilon is None else epsilon
        
        self.scale_factors = {} 
        self.mPF = {}
        self.PF = {}
        n_eqcons = 0
        
        self.func = problem_transition(func,self.lb, self.ub, self.n_objs, n_cons, n_eqcons, epsilon, logfile)
        
        self.use_cons = True if n_cons + n_eqcons > 0 else False
        
        self.set_greedy = set_greedy
        
        self.random_initialization()
        self.ref_point = torch.tensor(ref_point).to(device=self.device,dtype=torch.float32)
        self.bo_hv = Hypervolume(ref_point = -self.ref_point)
        
        #Ys = np.array([s['Y'] for s in self.samples])
        z_dir_tmp = self.scale_factors['z_dir']
        z_dir_tmp = np.maximum(z_dir_tmp,ref_point)
        
        self.normalize_hv = Hypervolume(ref_point = -torch.tensor(z_dir_tmp,device=self.device))
        
        #self.HV_trace = self.normalize_hv.compute(-torch.tensor(self.PF['Y'],device=self.device))
        self.HV_trace = self.compute_Q()
        
        root = TreeNode(self.samples, dims, self.normalize_hv, use_cons = self.use_cons, max_leaf_size = leaf_size, parent = None, node_id = 0, kernel_type = kernel_type, device = self.device, use_ard=self.use_ard)
        self.nodes.append( root )
        self.node_counter = 1
        
    
        
    def split_node(self,nodeid):
        assert self.nodes[nodeid].num_samples >= self.LEAF_SIZE
        
        lchild_data, rchild_data = self.nodes[nodeid].split()
        lchildid, rchildid = self.node_counter, self.node_counter + 1
        
        lchild = TreeNode(lchild_data, self.dims, self.normalize_hv, use_cons = self.use_cons, max_leaf_size = self.LEAF_SIZE, parent = nodeid, node_id = lchildid, kernel_type = self.kernel_type, device = self.device, use_ard=self.use_ard)
        self.nodes.append(lchild)
        
        rchild = TreeNode(rchild_data, self.dims, self.normalize_hv, use_cons = self.use_cons, max_leaf_size = self.LEAF_SIZE, parent = nodeid, node_id = rchildid, kernel_type = self.kernel_type, device = self.device, use_ard=self.use_ard)
        self.nodes.append(rchild)
        
        self.node_counter += 2
        
        numRound = (self.iterations-self.sample_counter)/self.batch_size/2
        self.nodes[nodeid].update_child(lchildid,rchildid,numRound = numRound)

        

    def get_scale_factors(self):
        if np.atleast_2d(self.PF['Y']).shape[0]> self.n_objs+self.n_cons:
            self.scale_factors['z_star'] = self.PF['Y'].min(axis = 0)
            self.scale_factors['z_dir'] = self.PF['Y'].max(axis = 0)
            if self.use_cons:
                self.scale_factors['c_star'] = self.PF['Cons'].min(axis = 0)
                self.scale_factors['c_dir'] = self.PF['Cons'].max(axis = 0)
        else:
            self.scale_factors['z_star'] = self.mPF['Y'].min(axis = 0)
            self.scale_factors['z_dir'] = self.mPF['Y'].max(axis = 0)
            if self.use_cons:
                self.scale_factors['c_star'] = self.mPF['Cons'].min(axis = 0)
                self.scale_factors['c_dir'] = self.mPF['Cons'].max(axis = 0)
        return self.scale_factors
    
    
    
    def update_PF(self,newYs,num_pareto=3,newCons=None):
        
        Y = newYs if len(self.mPF) == 0 else np.concatenate((self.mPF['Y'],newYs),axis=0)
        num_samples = len(Y)
        
        if self.use_cons:
            Cons = newCons if len(self.mPF) == 0 else np.concatenate((self.mPF['Cons'],newCons),axis=0)
            labels = np.zeros(num_samples)
            is_feasible = (Cons <= 0.0).all(axis=1)
            labels[is_feasible] = fast_non_dominated_sort(Y[is_feasible])
            max_label = labels.max()
            
            labels[np.bitwise_not(is_feasible)] = max_label + fast_non_dominated_sort( np.maximum(Cons[np.bitwise_not(is_feasible)],0.0))
            
            self.PF['Cons'] = Cons[labels==1]
            self.mPF['Cons'] = Cons[labels<=num_pareto]
            
        else:
            labels = fast_non_dominated_sort(Y)
        
        self.PF['Y'] = Y[labels==1]
        self.mPF['Y'] = Y[labels<=num_pareto]
        self.mPF['labels'] = labels[labels<=num_pareto]
        
        self.get_scale_factors()
        return 
    
    
    def evaluate_fun(self, sample):
        Ys, Cons = self.func.evaluate(sample)
        num_pnts = len(Ys)
        self.sample_counter += num_pnts
        xsample = np.atleast_2d(sample)
        newsamples = []
        for j in range(num_pnts):
            dic = {'X':xsample[j], 'Y':Ys[j], 'C':Cons[j]} if self.use_cons else {'X':xsample[j], 'Y':Ys[j], 'C':None}
            newsamples.append(dic)
            self.samples.append( dic )
        #print('Cons = ',Cons)
        self.update_PF(Ys, num_pareto=3, newCons=Cons)
        return newsamples
        
    def random_initialization(self):
        #latin hypercube sampling is used to generate init samples in search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = to_size(init_points, self.lb, self.ub)
        self.evaluate_fun(init_points)
        
        
    def compute_Q(self):
        if self.use_cons:
            if (self.PF['Cons']<=0).all():
                Q = self.normalize_hv.compute(-torch.tensor(self.PF['Y'],device=self.device))
            else:
                #scale_factors = self.get_scale_factors()
                c_scale = self.scale_factors['c_dir'] - self.scale_factors['c_star']
                CV = np.sum(np.maximum(self.PF['Cons'],0) / c_scale,axis=1)
                Q = -CV.min()
        else:
            Q = self.normalize_hv.compute(-torch.tensor(self.PF['Y'],device=self.device))
        return Q
    
        
    def update_recursive(self, samples, leaf_id, path):
        assert len(samples) > 0
        #self.get_scale_factors()
        c_scale = None if not self.use_cons else self.scale_factors['c_dir']-self.scale_factors['c_star']
        
        self.nodes[leaf_id].update(samples,None,None,c_scale)
        node_id = self.nodes[leaf_id].parent
        HV_old = self.HV_trace
        
        #self.HV_trace = self.normalize_hv.compute(-torch.tensor(self.PF['Y'],device=self.device))
        self.HV_trace = self.compute_Q()
        HVI = self.HV_trace - HV_old
        print('HVI = ', HVI)
        j = 1
        while node_id is not None: 
            reward = 1 if np.abs(HVI)>1e-6 else 0
            #reward = HVI
            self.nodes[node_id].update(samples,path[-j],reward,c_scale)
            node_id = self.nodes[node_id].parent
            j += 1
        

    def select(self):
        node_idx = 0
        path     = []
        nodelist = [0]

        while not self.nodes[node_idx].is_leaf():
            
            Action = self.nodes[node_idx].MAB.draw()
            print(self.nodes[node_idx].MAB.probabilityDistribution)
            path.append(Action)
            node_idx = self.nodes[node_idx].lchild if Action == 0 else self.nodes[node_idx].rchild
                
            nodelist.append(node_idx)
        
        print("Current node : ", node_idx )
        print("Path : ", path)
        if self.use_cons:
            PF_indicator = torch.all(self.nodes[node_idx].Cons<=0,dim=1)
            Ys = self.nodes[node_idx].Y[PF_indicator]
            if len(Ys)>0:
                hv = self.bo_hv.compute(-Ys)
            else:
                hv= -torch.maximum(self.nodes[node_idx].Cons,torch.tensor(0.0,device=self.device)).sum(dim=1).max()
        else:
            hv = self.bo_hv.compute(-self.nodes[node_idx].Y)
        print("Target node HV : ", hv)
        #print("Target node HV : ", self.bo_hv.compute(-self.nodes[node_idx].Y))
        return node_idx, path, nodelist
    
    
    def propose_samples_EPFI(self, leaf_idx, path, nodelist, num_samples=100000):
        
        if len(path) > 0:
            A = torch.tensor([self.nodes[j].split_hyperplane.clone().detach().cpu().numpy() if k==0 else -self.nodes[j].split_hyperplane.clone().detach().cpu().numpy() for j,k in zip(nodelist[:-1],path)],device = self.device)
            A = torch.atleast_2d(A)
        
        z_dir = self.scale_factors['z_dir']
        z_star = self.scale_factors['z_star'] - 0.1 * (self.scale_factors['z_dir']-self.scale_factors['z_star'])
        
        #normalized_Y = (self.mPF['Y']-z_star)/(z_dir-z_star)
        PF_shape = PF_surface_estimination(self.PF['Y'])
        
        cellidxes, optimize_cons = self.nodes[leaf_idx].batch_selection(self.scale_factors,self.batch_size,PF_shape)
        #optimize_cons = False
        directions = []
        PF_shape = PF_surface_estimination(self.PF['Y'])
        print('PF surface is ',PF_shape)
        for k in range(len(cellidxes)):
            if optimize_cons[k]:
                directions.append(None)
            else:
                z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                
                v0 = np.zeros(self.n_objs)
                v0[np.where(z<=0.2)] = 1.0*self.expansion_factor
                v0[np.where(z>=0.8)] = -1.0*self.expansion_factor
                    
                if PF_shape=='linear':
                    direction = 1./np.sqrt(2)+v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
                    #directions.append( -1./np.sqrt(self.n_objs) * np.ones(self.n_objs))
                elif PF_shape=='convex':
                    #z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                    direction = 1.2 - z
                    direction += v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
                else:
                    #z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                    direction = 0.2 + z
                    direction += v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
        
                
        GP_objs = self.nodes[leaf_idx].GP_objs
        
        z_star_tensor = torch.tensor(z_star,device=self.device)
        z_dir_tensor = torch.tensor(z_dir,device=self.device)
        
        if self.use_cons:
            GP_cons = self.nodes[leaf_idx].GP_cons
            
        proposed_X_np = np.zeros((self.batch_size,self.dims))
        for k in range(len(cellidxes)):
            cellidx = cellidxes[k]
            direction = torch.tensor(directions[k],device=self.device) if directions[k] is not None else None
            print('direction = ',direction)
            Y_ref = (self.nodes[leaf_idx].Y[cellidx] - z_star_tensor) / (z_dir_tensor - z_star_tensor)
            opt_cons = optimize_cons[k]
            def EIc(Xin):
                x = torch.atleast_2d(Xin)
                num = len(x)
                vals = -torch.inf * torch.ones(num).double()
                in_region = torch.zeros(num,device = self.device)< 1.0
                
                # x_1: num*(dim+1)  A: depth*(dim+1)    A * x_1: depth*num
                if len(path) > 0:
                    in_region[torch.any(A.matmul(torch.hstack((x,torch.ones((num,1),device=self.device))).T)>=0,dim=0)] = False
                
                # incell decision
                in_region[self.nodes[leaf_idx].kdtree.query(x.clone().detach().cpu().numpy(), eps=0, k=1)[1]!=cellidx] = False
                
                x_pre = torch.atleast_2d(x[in_region]).clone().detach()
                if opt_cons:
                    mu_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    sigma_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    
                    for j in range(len(GP_cons)):
                        mu_cons[:,j], sigma_cons[:,j] = GP_cons[j].predict(x_pre)
                    
                    PF = appro_normcdf(-mu_cons/sigma_cons).prod(dim=1)
                    vals[in_region] = PF.double()
                    
                    
                elif self.use_cons:
                    #print('Using constraints')
                    mu_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    sigma_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    
                    for j in range(len(GP_cons)):
                        mu_cons[:,j], sigma_cons[:,j] = GP_cons[j].predict(x_pre)
                        
                    mu_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    sigma_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    for k in range(self.n_objs):
                        mu_objs[:,k], sigma_objs[:,k] = GP_objs[k].predict(x_pre)
                        
                    mu_z = (direction * (mu_objs-Y_ref)).sum(dim=1)
                    sigma_z = torch.sqrt( (direction**2 * (sigma_objs**2)).sum(dim=1))
                    gamma = mu_z/sigma_z
                    
                    #PI = appro_normcdf(mu_z/sigma_z) #
                    EI = mu_z * appro_normcdf(gamma) + sigma_z * normpdf(gamma)
                    PF = appro_normcdf(-mu_cons/sigma_cons).prod(dim=1)
                    vals[in_region] = EI * PF
                    
                else:
                    mu_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    sigma_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    
                    for k in range(self.n_objs):
                        mu_objs[:,k], sigma_objs[:,k] = GP_objs[k].predict(x_pre)
                        
                    mu_z = (direction * (mu_objs-Y_ref)).sum(dim=1)
                    sigma_z = torch.sqrt( (direction**2 * (sigma_objs**2)).sum(dim=1))
                    gamma = mu_z/sigma_z
                    
                    EI = mu_z * appro_normcdf(gamma) + sigma_z * normpdf(gamma) 
                    vals[in_region] = EI 
                
                return vals
            
            cyc = 10
            num_sample = num_samples//cyc
            x0 = self.nodes[leaf_idx].X[cellidx].clone().detach().cpu().numpy()
            r = 0.3 * torch.std(self.nodes[leaf_idx].X,dim = 0).clone().detach().cpu().numpy() + self.sigma_x
            
            x_init = None
            while x_init is None:
                target_region = np.array([np.maximum(x0-r,self.lb), np.minimum(x0+r,self.ub)])
                
                x_sample = to_size(np.random.rand(num_sample, self.dims), target_region[0], target_region[1])
                EI_val = EIc(torch.tensor(x_sample,device=self.device))
                if EI_val.max() > -0.1:
                    x_init = x_sample[torch.argmax(EI_val)]
                    print('EI_init = ',EI_val.max().item())
                    break
                else:
                    r = 0.85 * r
                
            #print(target_region)
            proposed_X, acq_value = acq_min_msp(lambda x:-EIc(x), lambda x:-finite_diff(x,EIc), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
            
            proposed_X_np[k] = proposed_X.clone().detach().cpu().numpy()
            
        return proposed_X_np
    
    def propose_samples_PPFI(self, leaf_idx, path, nodelist, num_samples=100000):
        
        if len(path) > 0:
            A = torch.tensor([self.nodes[j].split_hyperplane.clone().detach().cpu().numpy() if k==0 else -self.nodes[j].split_hyperplane.clone().detach().cpu().numpy() for j,k in zip(nodelist[:-1],path)],device = self.device)
            A = torch.atleast_2d(A)
        
        z_dir = self.scale_factors['z_dir']
        z_star = self.scale_factors['z_star'] - 0.1 * (self.scale_factors['z_dir']-self.scale_factors['z_star'])
        
        #normalized_Y = (self.mPF['Y']-z_star)/(z_dir-z_star)
        PF_shape = PF_surface_estimination(self.PF['Y'])
        
        #PF_approximator = appeoximate_PF(normalized_Y,self.mPF['labels'])
        
        cellidxes, optimize_cons = self.nodes[leaf_idx].batch_selection(self.scale_factors,self.batch_size,PF_shape)
        #optimize_cons = False
        directions = []
        PF_shape = PF_surface_estimination(self.PF['Y'])
        print('PF surface is ',PF_shape)
        for k in range(len(cellidxes)):
            if optimize_cons[k]:
                directions.append(None)
            else:
                z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                
                v0 = np.zeros(self.n_objs)
                v0[np.where(z<=0.2)] = 1.0*self.expansion_factor
                v0[np.where(z>=0.8)] = -1.0*self.expansion_factor
                
                if PF_shape=='linear':
                    direction = 1./np.sqrt(2)+v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
                    #directions.append( -1./np.sqrt(self.n_objs) * np.ones(self.n_objs))
                elif PF_shape=='convex':
                    #z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                    direction = 1.2 - z
                    direction += v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
                else:
                    #z = (self.nodes[leaf_idx].Y[cellidxes[k]].clone().detach().cpu().numpy()-z_star)/(z_dir-z_star)
                    direction = 0.2 + z
                    direction += v0
                    direction = -direction / np.linalg.norm(direction)
                    directions.append(direction)
        
                
        GP_objs = self.nodes[leaf_idx].GP_objs
        
        z_star_tensor = torch.tensor(z_star,device=self.device)
        z_dir_tensor = torch.tensor(z_dir,device=self.device)
        
        if self.use_cons:
            GP_cons = self.nodes[leaf_idx].GP_cons
            
        proposed_X_np = np.zeros((self.batch_size,self.dims))
        for k in range(len(cellidxes)):
            cellidx = cellidxes[k]
            direction = torch.tensor(directions[k],device=self.device) if directions[k] is not None else None
            print('direction = ',direction)
            Y_ref = (self.nodes[leaf_idx].Y[cellidx] - z_star_tensor) / (z_dir_tensor - z_star_tensor)
            opt_cons = optimize_cons[k]
            def PIc(Xin):
                x = torch.atleast_2d(Xin)
                num = len(x)
                vals = -torch.inf * torch.ones(num).double()
                in_region = torch.zeros(num,device = self.device)< 1.0
                
                # x_1: num*(dim+1)  A: depth*(dim+1)    A * x_1: depth*num
                if len(path) > 0:
                    in_region[torch.any(A.matmul(torch.hstack((x,torch.ones((num,1),device=self.device))).T)>=0,dim=0)] = False
                
                # incell decision
                in_region[self.nodes[leaf_idx].kdtree.query(x.clone().detach().cpu().numpy(), eps=0, k=1)[1]!=cellidx] = False
                
                x_pre = torch.atleast_2d(x[in_region]).clone().detach()
                if opt_cons:
                    mu_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    sigma_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    
                    for j in range(len(GP_cons)):
                        mu_cons[:,j], sigma_cons[:,j] = GP_cons[j].predict(x_pre)
                    
                    PF = appro_normcdf(-mu_cons/sigma_cons).prod(dim=1)
                    vals[in_region] = PF.double()
                    
                    
                elif self.use_cons:
                    #print('Using constraints')
                    mu_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    sigma_cons = torch.zeros((len(x_pre),len(GP_cons)),device=self.device)
                    
                    for j in range(len(GP_cons)):
                        mu_cons[:,j], sigma_cons[:,j] = GP_cons[j].predict(x_pre)
                        
                    mu_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    sigma_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    for k in range(self.n_objs):
                        mu_objs[:,k], sigma_objs[:,k] = GP_objs[k].predict(x_pre)
                        
                    mu_z = (direction * (mu_objs-Y_ref)).sum(dim=1)
                    sigma_z = torch.sqrt( (direction**2 * (sigma_objs**2)).sum(dim=1))
                    PI = appro_normcdf(mu_z/sigma_z) 
                    PF = appro_normcdf(-mu_cons/sigma_cons).prod(dim=1)
                    if self.set_greedy:
                        wPI = appro_normcdf((Y_ref - mu_objs)/sigma_objs).prod(dim=1)
                        vals[in_region] = PI * PF * wPI
                    else:
                        vals[in_region] = PI * PF
                    
                else:
                    mu_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    sigma_objs = torch.zeros((len(x_pre),self.n_objs),device=self.device)
                    
                    for k in range(self.n_objs):
                        mu_objs[:,k], sigma_objs[:,k] = GP_objs[k].predict(x_pre)
                        
                    mu_z = (direction * (mu_objs-Y_ref)).sum(dim=1)
                    sigma_z = torch.sqrt( (direction**2 * (sigma_objs**2)).sum(dim=1))
                    PI = appro_normcdf(mu_z/sigma_z)
                    #gamma = mu_z/sigma_z
                    if self.set_greedy:
                        wPI = appro_normcdf((Y_ref - mu_objs)/sigma_objs).prod(dim=1)
                        vals[in_region] = PI * wPI
                    else:
                        vals[in_region] = PI
                return vals
            
            cyc = 10
            num_sample = num_samples//cyc
            x0 = self.nodes[leaf_idx].X[cellidx].clone().detach().cpu().numpy()
            r = 0.3 * torch.std(self.nodes[leaf_idx].X,dim = 0).clone().detach().cpu().numpy() + self.sigma_x
            
            x_init = None
            while x_init is None:
                target_region = np.array([np.maximum(x0-r,self.lb), np.minimum(x0+r,self.ub)])
                
                x_sample = to_size(np.random.rand(num_sample, self.dims), target_region[0], target_region[1])
                PI_val = PIc(torch.tensor(x_sample,device=self.device))
                if PI_val.max() > -0.1:
                    x_init = x_sample[torch.argmax(PI_val)]
                    print('PI_init = ',PI_val.max().item())
                    break
                else:
                    r = 0.85 * r
            
            proposed_X, acq_value = acq_min_msp(lambda x:-PIc(x), lambda x:-finite_diff(x,PIc), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
            
            proposed_X_np[k] = proposed_X.clone().detach().cpu().numpy()
            
        return proposed_X_np
    
    
    
    def search(self):
        if self.acq_func=='PPFI':
            AcqFunc = self.propose_samples_PPFI
        elif self.acq_func=='EPFI':
            AcqFunc = self.propose_samples_EPFI
        else:
            print('Invalid acquisition function! Setting to PPFI......')
            AcqFunc = self.propose_samples_PPFI
            
        for idx in range(self.iterations):
            print("")
            print("#"*20)
            print("Iteration:", idx)
            self.idx = idx
            
            leaf_idx, path, nodelist = self.select()
            xsample = AcqFunc( leaf_idx, path, nodelist )
            #xsample = self.propose_samples_PPFI( leaf_idx, path, nodelist )
            #xsample = self.propose_samples_EPFI( leaf_idx, path, nodelist )
            #print('xsample = ',xsample)
            samples = self.evaluate_fun( xsample )
            
            
            #update
            self.update_recursive(samples, leaf_idx, path)
            #split
            if self.nodes[leaf_idx].is_splittable():
                self.split_node(leaf_idx)
            
            print("Total samples:", len(self.samples) )
            if self.use_cons:
                PF_indicator = np.all(self.PF['Cons']<=0,axis=1)
                Ys = self.PF['Y'][PF_indicator]
                if len(Ys)>0:
                    hv = self.bo_hv.compute(-torch.tensor(Ys,device=self.device))
                else:
                    hv= -np.maximum(self.PF['Cons'],0.0).sum(axis=1).min()
            else:
                hv = self.bo_hv.compute(-torch.tensor(self.PF['Y'],device=self.device))
            
            print("Current HyperVolume:", hv)



def finite_diff(x_tensor,f,epslong=1e-6):
    with torch.no_grad():
        dims = len(x_tensor)
        delta = epslong*torch.eye(dims,device = x_tensor.device)
        ys = f(torch.cat((x_tensor + delta,x_tensor - delta),dim = 0))
        grad = (ys[:dims] - ys[dims:])/(2*epslong)
        #print('grad = ',grad)
    return grad

def appro_normcdf(x_tensor):
    #use Logistic function compute cdf sigma(1.702*x),
    #sigma(x) = 1/(1+exp(-x))
    return 1./(1+torch.exp(-1.702*x_tensor))

def normpdf(x_tensor):
    #1/sqrt(2*pi) = 0.39695
    return 0.39695*torch.exp(-0.5*x_tensor**2)
