import numpy as np
import torch
import scipy

import os
import h5py
import time
import copy

from span import Span


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.add_argument('--data_file', default='data.h5', type=str)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--max_iter_em_pretrain', default=1, type=int)
    parser.add_argument('--min_iter_adam_pretrain', default=3, type=int)
    parser.add_argument('--max_iter_adam_pretrain', default=300, type=int)
    parser.add_argument('--lr_adam', default=1e-2, type=float)
    parser.add_argument('--rel_tol_adam', default=1e-3, type=float)
    parser.add_argument('--max_iter_em_train', default=30, type=int)
    parser.add_argument('--min_iter_adam_train', default=10, type=int)
    parser.add_argument('--max_iter_adam_train', default=600, type=int)
    parser.add_argument('--output_file', default='predict.out')
    parser.add_argument('--device', default='cuda')


    args = parser.parse_args()

    
    ###########################################################
    ### Y [N by G] : gene matrix 
    ### label [N] : label
    ### rho [G by K]: cluster marker index matrix
    ### Z_neighbor_idx [N, num of neighbors], padding with -1
    ### batch_matrix [N, num of batches]
    ###########################################################    
    h5 = h5py.File(args.data_file, 'r')
    Y = np.array(h5['genes'])  #[n]
    label = np.array(h5['group']) if 'group' in h5.keys() else None
    rho = np.array(h5['rho'])
    Z_neighbor_idx = np.array(h5['Z_neighbor_idx'])
    batch_matrix = np.array(h5['batch_matrix']) if 'batch_matrix' in h5.keys() else None
    h5.close()
    
    if batch_matrix is not None:
        n_cov = batch_matrix.shape[1]
    else:
        n_cov = 0
        
    S = Y.sum(axis = 1, keepdims=True)
    S = S/np.mean(S)    
        
    model = Span(Y.shape[0], rho.shape[0], rho.shape[1], rho, Y, S, Z_neighbor_idx,
                    n_cov  = n_cov, cov_matrix = batch_matrix, 
                    batch_size = args.batch_size).to(args.device)
        
    model.pre_train_model(max_iter_em = args.max_iter_em_pretrain, min_iter_adam = args.min_iter_adam_pretrain, 
                          max_iter_adam = args.max_iter_adam_pretrain, lr_adam = args.lr_adam,
                            rel_tol_adam = args.rel_tol_adam, 
                          y_true = label)
    
    model.train_model(max_iter_em = args.max_iter_em_train, min_iter_adam = args.min_iter_adam_train, 
                  max_iter_adam = args.max_iter_adam_train, lr_adam = args.lr_adam, 
                  y_true = label)
    
    np.savetxt(args.output_file, model.Z_current, delimiter=",")
