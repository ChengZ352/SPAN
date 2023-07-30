import os
import numpy as np
import scipy
from scipy.special import logsumexp
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.distributions import Dirichlet
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from layers import NBModule, SPModule


class Span(nn.Module):
    def __init__(self, n_samples, n_genes, n_label, rho, Y, S, 
                 Z_neighbor_idx, Phi=3,
                 n_cov  = 0, cov_matrix = None, batch_size = -1,
                 B = 10, min_log_delta = 1e-6, random_b_g_0  = False,
                 device = 'cuda',
        ):
        super(Span, self).__init__()
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_labels = n_label
        self.batch_size = batch_size
        self.B = B
        self.Yt = torch.Tensor(Y).to(device)
        self.St = torch.Tensor(S).to(device)
        
        col_means = np.asarray(np.mean(Y, 0)).ravel()  # (g)
        col_means_mu, col_means_std = np.mean(col_means), np.std(col_means, ddof=1)
        col_means_normalized = torch.Tensor((col_means - col_means_mu) / col_means_std)
        
        # compute basis means for phi - shape (B)
        basis_means = np.linspace(np.min(Y), np.max(Y), B)  # (B)
        
        self.Z_neighbor_idx = Z_neighbor_idx
        
        # init parameter for SPModule    
        self.pt = np.ones(self.n_labels + self.n_labels * (self.n_labels - 1) //2 ) * Phi
        
        self.module = NBModule(
            n_genes=self.n_genes,
            n_labels = self.n_labels,
            rho = torch.tensor(rho),
            basis_means=basis_means,
            b_g_0=col_means_normalized,
            random_b_g_0 = random_b_g_0,
            n_cov = n_cov,
            B=B)
        
        self.spm = SPModule(self.n_labels, self.pt[:self.n_labels], self.pt[self.n_labels:], device)
        
        self.min_log_delta = min_log_delta
        if cov_matrix is None:
            self.cov_matrix = None
        else:
            self.cov_matrix = torch.Tensor(cov_matrix).to(device)
        self.device = device
            
    def count_pix_neighbor(self, Z):
        n, m = self.Z_neighbor_idx.shape
        Z = Z.astype(int)
        Z_neighbor_count = np.zeros([self.n_samples, self.n_labels])

        for i in range(n):
            for j in range(m):
                if(self.Z_neighbor_idx[i, j] != -1):
                    Z_neighbor_count[i,  Z[self.Z_neighbor_idx[i, j]] ] +=1
               
        return Z_neighbor_count #[n, k]

    def logphi_m(self, alpha, beta, z, z_neighbor_count, k=2):
        beta_m = np.zeros([k,k])
        temp = 0
        for i in range(k):
            temp1 = temp + k-i-1
            beta_m[i, i+1:] = beta[temp:temp1]
            beta_m[i+1:, i] = beta[temp:temp1]
            temp = temp1
                
        log_phi_k = np.zeros(z.shape+(k,))
        beta_mu = z_neighbor_count.dot(beta_m) #[n, k]
        alpha_beta_mu = alpha[np.newaxis,:] - beta_mu
        denominator = logsumexp(alpha_beta_mu, axis = 1, keepdims = True) #[n, 1]
        result = alpha_beta_mu - denominator
        return result 

 
    
    def updateZ(self, optim_adam_spm, lr_z=3e-4, max_iter = 500):
        Z_neighbor_count = self.count_pix_neighbor(self.Z_current)
       
        loss_temp = []
        
        num = self.Yt.shape[0]
        sample_indices = np.arange(num)
        if (self.batch_size <= 0):
            self.batch_size = num
        num_batch = int(math.ceil(1.0*num/self.batch_size))
        
        temp = []
        
        
        for ii in range(1, max_iter+1):
            loss_val = 0
            for batch_idx in range(num_batch):
                
                batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]

                Z_current_batch = torch.Tensor(self.Z_current).to(self.device)[batch_indices]
                Z_neighbor_count_batch = torch.Tensor(Z_neighbor_count).to(self.device)[batch_indices]
                r_batch = self.r[batch_indices]
                
                
                loss = self.spm.loss(Z_current_batch, Z_neighbor_count_batch, r_batch) 
            loss_val = loss_val + loss
            loss_val = loss_val / num
            self.spm.zero_grad()
            loss_val.backward()
            optim_adam_spm.step()
            loss_temp.append(loss_val)
            
            if(ii>10):
                old = loss_temp.pop(0)
                if( (old-loss_val)/loss_val < 1e-3):
                    break

        alpha = self.spm.alpha.detach().cpu().numpy()
        beta = self.spm.beta.detach().cpu().numpy()
        self.pt = np.concatenate([alpha, beta])

        
        self.eval()
        logphi_mk = self.logphi_m(alpha, beta, self.Z_current, Z_neighbor_count, self.n_labels) #[n,k]

        sample_indices = np.arange(num)
        temp = []
        for batch_idx in range(num_batch):
            batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]
            Yt_batch = self.Yt[batch_indices]
            St_batch = self.St[batch_indices]
            cov_batch = None if self.cov_matrix is None else self.cov_matrix[batch_indices]

            dic = self.module.generate(Yt_batch, St_batch, cov_matrix = cov_batch)
            p_x_c_ = dic["p_x_c"].detach().cpu().numpy()
            temp.append(p_x_c_)
        p_x_c_ = np.concatenate(temp, axis = 0)
        
        loglh = logphi_mk + p_x_c_ 
        self.Z_current = np.argmax(loglh, axis = -1)
        return loss

    
    def encode_batch(self, y_true = None):
        self.eval()
        num = self.Yt.shape[0]
        sample_indices = np.arange(num)
        if (self.batch_size==-1):
            self.batch_size = num
        num_batch = int(math.ceil(1.0*num/self.batch_size))
        
        temp = []
        for batch_idx in range(num_batch):
            batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]
            Yt_batch = self.Yt[batch_indices]
            St_batch = self.St[batch_indices]
            cov_batch = None if self.cov_matrix is None else self.cov_matrix[batch_indices]

            dic = self.module.generate(Yt_batch, St_batch, cov_matrix = cov_batch, pre_train = True)
            temp.append(dic["gamma"].detach().cpu().numpy())
            
        
        self.Z_current = np.argmax(np.concatenate(temp, axis = 0), 1)


    
    def pre_train_model(self, max_iter_em = 10, min_iter_adam = 60, max_iter_adam = 300, rel_tol_adam = 1e-3, lr_adam = 1e-2, y_true = None):
        
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr_adam, amsgrad=True)

        num = self.Yt.shape[0]
        sample_indices = np.arange(num)
        if (self.batch_size==-1):
            self.batch_size = num
        num_batch = int(math.ceil(1.0*num/self.batch_size))
        
        z_prev = None
        
        for i in range(max_iter_em):
            
            self.eval()
            gamma_temp = []
            for batch_idx in range(num_batch):
                batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]

                Yt_batch = self.Yt[batch_indices]
                St_batch = self.St[batch_indices]
                cov_batch = None if self.cov_matrix is None else self.cov_matrix[batch_indices]

                dic = self.module.generate(Yt_batch, St_batch, cov_matrix = cov_batch, pre_train = True)
                gamma_fixed = dic["gamma"].detach()
                gamma_fixed = gamma_fixed+1e-6
                gamma_fixed = gamma_fixed/torch.sum(gamma_fixed, axis = 1, keepdims = True)

                if(i==0):
                    d = Dirichlet(torch.tensor([1/self.n_labels]*self.n_labels))
                    gamma_fixed = d.sample([batch_indices.shape[0]]).to(self.device)
                    
                gamma_temp.append(gamma_fixed)
                
            gamma_fixed = torch.cat(gamma_temp, axis = 0)

            loss_diff = rel_tol_adam + 1
            
            j = 0

            self.train()
            loss_val = 1
            while( (j <= min_iter_adam) or ( (j < max_iter_adam)  and  (loss_diff > rel_tol_adam ) ) ):
                j += 1
                loss_val_new = 0

                for batch_idx in range(num_batch):
                    batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]

                    Yt_batch = self.Yt[batch_indices]
                    St_batch = self.St[batch_indices]
                    cov_batch = None if self.cov_matrix is None else self.cov_matrix[batch_indices]
                    gamma_batch = gamma_fixed[batch_indices]
                
                    dic = self.module.generate(Yt_batch, St_batch, cov_matrix = cov_batch, pre_train = True)
                    p_x_c = dic["p_x_c"]
                    loss, _, _ = self.module.loss(p_x_c, gamma_batch)
                    loss_val_new += loss*Yt_batch.shape[0]
                    
                loss_val_new = loss_val_new/self.n_samples
                self.zero_grad()
                loss_val_new.backward()
                clip_grad_norm_(self.parameters(), 1-1e-5, 2)
                optim_adam.step()
                
                with torch.no_grad():
                    self.module.delta_log_scale.clamp_(self.min_log_delta)
                

                loss_val_new = loss_val_new.item()
                loss_diff = -(loss_val_new - loss_val) / np.abs(loss_val) 
                loss_val = loss_val_new
                
            print('epoch {} loss: {}'.format(i,loss_val))
            self.encode_batch(y_true)
 
            if (y_true is not None):
                print("pretrain acc, ".format(i),  np.mean(y_true == self.Z_current ) )
            
            if(z_prev is not None):
                print('delta, {}'.format(np.mean(z_prev != self.Z_current)) )
                if( np.mean(z_prev != self.Z_current) < 0.001):
                    break
                
            z_prev = self.Z_current
        
        print('')
    
    
    def train_model(self, min_iter_em = 2, max_iter_em = 20, min_iter_adam = 100, max_iter_adam = 600, rel_tol_adam = 1e-4, lr_adam = 1e-2, lr_z = 3e-4,  y_true = None):
        
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr_adam, amsgrad=True)
        optim_adam_spm = optim.Adam(filter(lambda p: p.requires_grad, self.spm.parameters()), lr=lr_z, amsgrad=True)
      
        num = self.Yt.shape[0]
        sample_indices = np.arange(num)
        if (self.batch_size==-1):
            self.batch_size = num
        num_batch = int(math.ceil(1.0*num/self.batch_size))
    
        z_prev = None
        self.train()   
        for i in range(max_iter_em):  
            loss_diff = rel_tol_adam + 1
            j = 0
        
            #convert Z into one_hot Tensor
            temp  = np.zeros((self.Z_current.shape[0], self.n_labels))
            temp[np.arange(self.Z_current.shape[0]),self.Z_current] = 1
            self.r = torch.Tensor(temp).to(self.device)
            
            loss_val = 1
            
            while( (j <= min_iter_adam) or ( (j < max_iter_adam)  and  (loss_diff > rel_tol_adam ) ) ):
                j += 1
                loss_val_new = 0
                for batch_idx in range(num_batch):
                    batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]
                    Yt_batch = self.Yt[batch_indices]
                    St_batch = self.St[batch_indices]
                    cov_batch = None if self.cov_matrix is None else self.cov_matrix[batch_indices]
                    r_batch = self.r[batch_indices]
                
                    dic = self.module.generate(Yt_batch, St_batch, cov_matrix = cov_batch)
                    p_x_c = dic["p_x_c"]
                    loss, _, _ = self.module.loss(p_x_c, r_batch)
                    loss_val_new += loss*Yt_batch.shape[0]
                    
                loss_val_new = loss_val_new/self.n_samples
                self.zero_grad()
                loss_val_new.backward()#retain_graph=True
                clip_grad_norm_(self.parameters(), 1-1e-5, 2)
                optim_adam.step()
                
                with torch.no_grad():
                    self.module.delta_log_scale.clamp_(self.min_log_delta)
                
                loss_val_new = loss_val_new.item()
                loss_diff = -(loss_val_new - loss_val) / np.abs(loss_val)
                loss_val = loss_val_new
                
            
            loss_s = self.updateZ(optim_adam_spm, lr_z)
            
            if (y_true is not None):
                print("epoch {} acc, ".format(i),  np.mean(y_true == self.Z_current ) )
            else:
                print("epoch {}".format(i))

            if(z_prev is not None):
                if( np.mean(z_prev != self.Z_current) < 0.001):
                    break
    
            z_prev = self.Z_current



