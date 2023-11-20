import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from utils import sparse_mx_to_torch_sparse_tensor

def aug_normalized_adj_lap_sharpen_trick(adj, adj_temp):
    '''See our paper'''
    nodenum = adj.shape[0]
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)) + 2*np.ones([nodenum, 1])
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_temp).dot(d_mat_inv_sqrt).tocoo()
    return res

def get_random_sys_nor_adj(nodenum, k):
    ''' 
       Like the GCN model, but a random adj with sys_normalied
       Return: adj_sys_nor
    '''
    A = nx.adjacency_matrix(nx.random_regular_graph(d=k, n=nodenum))
    I = sp.eye(nodenum)
    A_temp = 2*I - A
    adj_sym_nor = aug_normalized_adj_lap_sharpen_trick(A, A_temp)
    adj_sym_nor = sparse_mx_to_torch_sparse_tensor(adj_sym_nor).float()
    return adj_sym_nor

# def get_inv_lap_cost(X, L):
#     return -1*get_laplace_cost(X, L)

def get_lazy_random_walk(A_sym_nor, I_N, beta):
    """ Lazy random walk can be seen a variant of graph convolution by changing waiting rate:
            lazy_A = beta*A +(1-beta)*I
    """ 
    lazy_A = beta*A_sym_nor + (1-beta)*I_N
    return lazy_A

def my_get_adj_org(dataset_str="cora"):
    # get not normalized adj
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    _, _, _, _, _, _, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj