import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal

from nb import NegativeBinomial


LOWER_BOUND = 1e-10
THETA_LOWER_BOUND = 1e-20
B = 10


class NBModule(nn.Module):
    def __init__(self, n_genes, n_labels, 
                 rho, basis_means,
                 b_g_0  = None, random_b_g_0 = False,
                 n_cov = 0):
        
        super(NBModule, self).__init__()
        
        self.n_genes = n_genes
        self.n_labels = n_labels
        self.n_cov = n_cov

        self.register_buffer("rho", rho)

        # perform all other initializations
        dirichlet_concentration = torch.tensor([1e-2] * self.n_labels)
        self.register_buffer("dirichlet_concentration", dirichlet_concentration)
        self.shrinkage = True
        if b_g_0 is None or random_b_g_0 is True:
            self.b_g_0 = torch.nn.Parameter(torch.randn(n_genes))
        else:
            self.b_g_0 = torch.nn.Parameter(b_g_0)

        # compute theta for pre_train
        self.theta_logit = torch.nn.Parameter( self.truncated_normal_(torch.empty([self.n_labels])) )

        # compute delta (cell type specific overexpression parameter)
        # will be clamped by callback during training
        self.delta_log = torch.nn.Parameter(
            nn.init.kaiming_normal_(torch.empty([self.n_genes, self.n_labels]), mode='fan_out', nonlinearity='relu')
        )

        # shrinkage prior on delta
        self.delta_log_mean = torch.nn.Parameter( torch.zeros( 1, ) )
        self.delta_log_scale = torch.nn.Parameter( torch.ones( 1, ) )

        self.log_a = torch.nn.Parameter(torch.zeros(B))

        if self.n_cov == 0:
            self.beta = None
        else:
            self.beta = torch.nn.Parameter( torch.zeros(self.n_genes, self.n_cov) )  # (g, p)

        self.register_buffer("basis_means", torch.tensor(basis_means))
   
    def truncated_normal_(self, tensor,mean=0,std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
    
    def generate(self, x, size_factor, cov_matrix=None, pre_train = False):
        # x has shape (n, g)
        delta = torch.exp(self.delta_log)  # (g, c)
        n_samples = x.shape[0]
        if pre_train:
            theta_log = F.log_softmax(self.theta_logit, dim=-1)  # (c)

        base_mean = torch.log(size_factor)  # (n, 1)
        base_mean = base_mean.unsqueeze(-1).expand( n_samples, self.n_genes, self.n_labels )  # (n, g, c)

        # compute beta (covariate coefficent)
        # cov_matrix (n,p)
        if cov_matrix is not None:
            covariates = torch.einsum("np,gp->gn", cov_matrix, self.beta)  # (g, n)
            covariates = torch.transpose(covariates, 0, 1).unsqueeze(-1)  # (n, g, 1)
            covariates = covariates.expand(n_samples, self.n_genes, self.n_labels)
            base_mean = base_mean + covariates

        # base gene expression
        b_g_0 = self.b_g_0.unsqueeze(-1).expand(n_samples, self.n_genes, self.n_labels)
        delta_rho = delta * self.rho
        delta_rho = delta_rho.expand(n_samples, self.n_genes, self.n_labels)  # (n, g, c)
        
        log_mu_ngc = base_mean + delta_rho + b_g_0
        mu_ngc = torch.exp(log_mu_ngc)  # (n, g, c)

        a = torch.exp(self.log_a)  # (B)
        a = a.expand(n_samples, self.n_genes, self.n_labels, B)
        b_init = 2 * ((self.basis_means[1] - self.basis_means[0]) ** 2)
        b = torch.exp(torch.ones(B, device=x.device) * (-torch.log(b_init)))  # (B)
        b = b.expand(n_samples, self.n_genes, self.n_labels, B)
        mu_ngcb = mu_ngc.unsqueeze(-1).expand( n_samples, self.n_genes, self.n_labels, B )  # (n, g, c, B)
        basis_means = self.basis_means.expand( n_samples, self.n_genes, self.n_labels, B )  # (n, g, c, B)
        phi = (  # (n, g, c)
            torch.sum(a * torch.exp(-b * torch.square(mu_ngcb - basis_means)), 3)
            + LOWER_BOUND
        )

        # compute gamma
        nb_pdf = NegativeBinomial(mu=mu_ngc, theta=phi)
        x_ = x.unsqueeze(-1).expand(n_samples, self.n_genes, self.n_labels)
        x_log_prob_raw = nb_pdf.log_prob(x_)  # (n, g, c)
        p_x_c = torch.sum(x_log_prob_raw, 1) #+ theta_log  # (n, c)
        gamma = None
        
        if pre_train:
            theta_log = theta_log.expand(n_samples, self.n_labels)
            p_x_c = p_x_c + theta_log
            normalizer_over_c = torch.logsumexp(p_x_c, 1)
            normalizer_over_c = normalizer_over_c.unsqueeze(-1).expand( n_samples, self.n_labels )
            gamma = torch.exp(p_x_c - normalizer_over_c)  # (n, c)
            
        return dict(
            mu=mu_ngc,
            phi=phi,
            gamma=gamma,
            p_x_c=p_x_c,)

    def loss( self, p_x_c, gamma, pre_train = False):
        
        q_per_cell = torch.sum(gamma * -p_x_c, 1)
        n_samples = p_x_c.shape[0]
        # third term is log prob of prior terms in Q
        if pre_train:
            theta_log = F.log_softmax(self.theta_logit, dim=-1)
            theta_log_prior = Dirichlet(self.dirichlet_concentration)
            theta_log_prob = -theta_log_prior.log_prob(
                torch.exp(theta_log) + THETA_LOWER_BOUND
            )
            prior_log_prob = theta_log_prob
        else:
            prior_log_prob = 0 
            
        delta_log_prior = Normal(
            self.delta_log_mean, self.delta_log_scale
        )
        delta_log_prob = torch.masked_select(
            delta_log_prior.log_prob(self.delta_log), (self.rho > 0)
        )
        prior_log_prob += -torch.sum(delta_log_prob)

        loss = (torch.mean(q_per_cell) * n_samples + prior_log_prob) / n_samples

        return  loss, q_per_cell, prior_log_prob


class SPModule(nn.Module):
   
    def __init__(self, k, alpha_init, beta_init, device):
        super(SPModule, self).__init__()
        self.k = k
        self.alpha = torch.nn.Parameter(torch.Tensor(alpha_init))
        self.beta = torch.nn.Parameter(torch.Tensor(beta_init))
        self.device  = device

    def loss(self, z, z_neighbor_count, r):
        #Convert upper diagonal to Symmetric matrix, with diagnoal = 0
        self.beta_m = torch.Tensor(np.zeros([self.k, self.k])).to(self.device)
        temp = 0
        for i in range(self.k):
            temp1 = temp + self.k-i-1
            self.beta_m[i, i+1:] = self.beta[temp:temp1]
            self.beta_m[i+1:, i] = self.beta[temp:temp1]
            temp = temp1
        
        # \sum_{k!=l} beta_kl * mu_i(l)
        beta_mu = torch.mm(z_neighbor_count, self.beta_m) #[n,k]
        alpha_beta_mu = self.alpha.unsqueeze(0)- beta_mu #[n,k] 
        denominator = torch.logsumexp(alpha_beta_mu, dim=1, keepdim=True,)
        result = alpha_beta_mu - denominator
        
        return - (result*r).sum()
            
         

    