# 
from .GP import train_gp
import numpy as np
from scipy.spatial import KDTree

import torch
from scipy.special import comb
from .utils import fast_non_dominated_sort
from .approximate_PF import cluster_fx
from .GradientBandits import GradientBandits


class TreeNode(object):
    
    
    def __init__(self, samples, dims, bo_hv, use_cons = False, max_leaf_size = 20, parent = None, node_id = 0, kernel_type = "rbf", device = 'cpu', use_ard = False):
        
        self.dims          = dims
        
        self.num_samples   = len(samples)
            
        self.max_leaf_size = max_leaf_size
        self.gp_kernel     = kernel_type
        
        self.parent        = parent        
        self.lchild        = None
        self.rchild        = None
        self.split_hyperplane = None
        
        self.X             = torch.tensor([s['X'] for s in samples],device=device)
        self.Y             = torch.tensor([s['Y'] for s in samples],device=device)
        
        self.n_objs = self.Y.shape[1]
        
        if use_cons:
            self.Cons      = torch.tensor([s['C'] for s in samples],device=device)
        else:
            self.Cons      = None
        
        self.use_ard       = use_ard
        self.use_cons      = use_cons
        self.device        = device
        
        self.kdtree        = KDTree(self.X.clone().detach().cpu().numpy())  
        self.id            = node_id
        self.MAB           = None
        
                

    
    def is_splittable(self):
        return self.num_samples >= self.max_leaf_size
        
    def is_root(self):
        return self.parent == None
        
    def is_leaf(self):
        return self.lchild == None and self.rchild == None
    
    
    def batch_selection(self,scale_factors,batch_size,PF_shape):
        
        assert self.is_leaf()
        
        if self.use_cons:
            c_scale = torch.tensor(scale_factors['c_dir']-scale_factors['c_star'],device=self.device)
            normalized_Cons = self.Cons / c_scale
            normalized_C = normalized_Cons.clone().detach().cpu().numpy()
            
        z_star = torch.tensor(scale_factors['z_star'],device=self.device)
        z_dir = torch.tensor(scale_factors['z_dir'],device=self.device)
        z_star = z_star - 0.1 * (z_dir-z_star)
        
        normalized_Y = (self.Y-z_star)/(z_dir-z_star)
        
        #print('Cons = ',self.Cons)
        
        self.GP_objs, self.GP_cons = train_gp(self.X,normalized_Y,self.Cons,use_ard=self.use_ard, num_steps=200,kernel_type = self.gp_kernel,bounds = torch.tensor([[0]*self.dims,[1]*self.dims], device=self.device))
        
        z = normalized_Y.clone().detach().cpu().numpy()
        labels = self.rank_samples() #fast_non_dominated_sort(z)
        H = 4
        num_ref = int(comb(H + self.n_objs-1, self.n_objs-1))
        
        PF_cnt = 0
        PF_tag = 0
        while PF_cnt<= max(batch_size,num_ref):
            PF_tag += 1
            PF_cnt += np.sum([1 for label in labels if label==PF_tag])
        
        select_flag = np.zeros(self.num_samples)>1
        candidate_indicator = labels <= PF_tag
        normaled_C = normalized_C[candidate_indicator] if self.use_cons else None
        select_flag[candidate_indicator] = cluster_fx(z[candidate_indicator],normaled_C,batch_size,PF_shape)
        
        self.selected_cells = np.arange(self.num_samples)[select_flag]
        
        optimize_cons = [] 
        for cell in self.selected_cells:
            if self.use_cons and np.maximum(normalized_C[cell],0).sum() >= 0.1:
                optimize_cons.append(True)
            else:
                optimize_cons.append(False)
        return self.selected_cells, optimize_cons#, directions
    
    
    def update(self, samples, j, reward, c_scale = None):
        assert len(samples) > 0
        for newsample in samples:
            
            if self.num_samples == 0:
                self.X = torch.atleast_2d(torch.tensor(newsample['X'],device=self.device))
                self.Y = torch.atleast_2d(torch.tensor(newsample['Y'],device=self.device))
                if self.use_cons:
                    self.Cons = torch.atleast_2d(torch.tensor(newsample['C'],device=self.device))
            else:
                self.X = torch.vstack((self.X, torch.atleast_2d(torch.tensor(newsample['X'],device=self.device))))
                self.Y = torch.vstack((self.Y, torch.atleast_2d(torch.tensor(newsample['Y'],device=self.device))))
                if self.use_cons:
                    self.Cons = torch.vstack((self.Cons, torch.atleast_2d(torch.tensor(newsample['C'],device=self.device))))
            
        self.num_samples += len(samples)
        
        if not self.is_leaf():
            self.MAB.update_reward(reward)
        #self.update_Q(bo_hv,c_scale)
        
        if self.is_leaf():
            self.kdtree = KDTree(self.X.clone().detach().cpu().numpy()) 
            
        return
        
    
    def rank_samples(self):
        if self.use_cons:
            self.labels = np.zeros(self.num_samples)
            is_feasible = (self.Cons <= 0.0).all(dim=1)
            is_feasible_np = is_feasible.clone().detach().cpu().numpy()
            
            self.labels[is_feasible_np] = fast_non_dominated_sort(self.Y[is_feasible].clone().detach().cpu().numpy())
            max_label = self.labels.max()
            
            self.labels[np.bitwise_not(is_feasible_np)] = max_label + fast_non_dominated_sort((self.Cons[torch.bitwise_not(is_feasible)]).maximum(torch.tensor(0.0)).clone().detach().cpu().numpy())
        else:
            self.labels = fast_non_dominated_sort(self.Y.clone().detach().cpu().numpy())
        
        return self.labels
    
        
    def split(self):
        assert self.num_samples >= 2
        self.rank_samples()
        centroids = kMeans(self.X.clone().detach().cpu().numpy(), self.labels)
        centroids = torch.tensor(centroids,device=self.device)
        A = centroids[0]-centroids[1]
        A = A/torch.linalg.norm(A,2)
        b = torch.dot(A,(centroids[0]+centroids[1])/2)
        self.split_hyperplane = torch.cat((A,-torch.atleast_1d(b)))
        is_lchild = torch.hstack((self.X,torch.ones((self.num_samples,1),device=self.device))).matmul(self.split_hyperplane) <= 0
        
        if self.use_cons:
            lchild_data = [{'X': s, 'Y': t, 'C': u} for s,t,u in zip(self.X[is_lchild].clone().detach().cpu().numpy(),self.Y[is_lchild].clone().detach().cpu().numpy(),self.Cons[is_lchild].clone().detach().cpu().numpy())]
            rchild_data = [{'X': s, 'Y': t, 'C': u} for s,t,u in zip(self.X[torch.bitwise_not(is_lchild)].clone().detach().cpu().numpy(),self.Y[torch.bitwise_not(is_lchild)].clone().detach().cpu().numpy(),self.Cons[torch.bitwise_not(is_lchild)].clone().detach().cpu().numpy())]
        else:
            lchild_data = [{'X': s, 'Y': t, 'C': None} for s,t in zip(self.X[is_lchild].clone().detach().cpu().numpy(),self.Y[is_lchild].clone().detach().cpu().numpy())]
            rchild_data = [{'X': s, 'Y': t, 'C': None} for s,t in zip(self.X[torch.bitwise_not(is_lchild)].clone().detach().cpu().numpy(),self.Y[torch.bitwise_not(is_lchild)].clone().detach().cpu().numpy())]
        
        #del(self.X,self.Y,self.Cons)
        assert len( lchild_data ) + len( rchild_data ) ==  self.num_samples
        assert len( lchild_data ) > 0 
        assert len( rchild_data ) > 0 
        #print('lchild_data = ',lchild_data)
        return lchild_data, rchild_data
    
    def update_child(self,lchild,rchild,numRound):
        self.lchild = lchild
        self.rchild = rchild
        self.MAB = GradientBandits()
        #self.MAB = EXP3()
        return






def kMeans(X, Y, k = 2, max_iter = 20):
    
    w = 0.25
    npnts, dims = X.shape
    centroids = np.zeros((k,dims))
    Y_centroids = np.zeros(k)
    x_weights = np.std(X,axis=0)
    weights = np.std(Y)
    
    weighted_dists = np.zeros((npnts,npnts))
    dists = np.zeros((npnts,npnts))
    f_dists = np.zeros((npnts,npnts))
    for j in range(npnts):
        dists[j] = np.linalg.norm((X-X[j,:])/x_weights, ord=2, axis=1) 
        f_dists[j] = np.abs((Y-Y[j])/weights)
        
    weighted_dists = w * dists/np.sqrt(dims) + (1-w) * f_dists
    idxes = weighted_dists.argmax()
    idx0 = idxes//npnts
    idx1 = idxes%npnts
    #print(np.array([X[idx0],X[idx1]]))
    centroids[:2,:]=np.array([X[idx0],X[idx1]])
    Y_centroids[:2] = np.array([Y[idx0],Y[idx1]])
    
    
    for j in range(k-2):
        weighted_dists = np.zeros((j+1,npnts))
        for i in range(j+1):
            dists[i] = w * np.linalg.norm((X-centroids[i,:])/x_weights, ord=2, axis=1)/np.sqrt(dims) + (1-w) * np.abs((Y-Y_centroids[i])/weights)
        minval, _ = dists.min(axis=0)
        centroids[j+1] = X[minval.argmax()]
        Y_centroids[j+1] = Y[minval.argmax()]
        
    weighted_dists = np.zeros((k,npnts))
    for i in range(max_iter):
        centroids_bk = np.array(centroids)
        for j in range(k):
            weighted_dists[j] = w * np.linalg.norm((X-centroids[j,:])/x_weights, ord=2, axis=1)/np.sqrt(dims) + (1-w) * np.abs((Y-Y_centroids[j])/weights)
        clusters = weighted_dists.argmin(axis = 0)
        
        for j in range(k):
            centroids[j] = np.mean(X[clusters==j],axis=0)
        max_variation = abs(centroids_bk - centroids).max()
        if max_variation<=1e-4:
            break
    return centroids


