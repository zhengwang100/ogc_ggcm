import argparse
import scipy.sparse as sp
import torch
from utils import load_citation, sparse_mx_to_torch_sparse_tensor, use_cuda
from ggcm_utils import get_random_sys_nor_adj, get_lazy_random_walk, my_get_adj_org
from eval_features_with_split import eval_classify
from args_ggcm import get_citation_args

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora", choices=["cora", "citeseer", "pubmed"], help='Dataset to use.')
args, _ = parser.parse_known_args()

args = get_citation_args(args.dataset)
device = use_cuda()

def GGCM(features, A_hat, alpha, neg_avg_edge_num):
    """
       Graph Structure Preserving Graph Convolution Multi-scale Version, GGCM.
       GGCM sum the embedding results (via GC + IGC) at all iterations, as:
           U = alpha*X + (1-alpha)*\sum_{t=1}^{k} [(U^{(t)}_{smo} + U^{(t)}_{sharp})/2].
       Args:
           features: original node attributes with shape [n, d]
           A_hat: normalized adjacency matrix of the graph with shape (n, n) 
           alpha: hyper-parameter for alpha*X + (1-alpha)...
           neg_avg_edge_num: negtive sampling rate for a negative (i.e., cannot link) graph 
        Return: embedds the final node embeddings with shape [n, d]
    """
    beta = 1.0
    beta_neg = 1.0
    K = args.degree
    X = features.clone()
    temp_sum = torch.zeros_like(features)
    I_N = sparse_mx_to_torch_sparse_tensor(sp.eye(features.shape[0])).float().to(device)
    for _ in range(K):
        # lazy graph convolution (LGC)
        pre_features = features
        lazy_A = get_lazy_random_walk(A_hat, I_N, beta=beta)
        features = torch.spmm(lazy_A, pre_features)
        # inverse graph convlution (IGC), lazy version
        neg_A_hat = get_random_sys_nor_adj(features.shape[0], neg_avg_edge_num).to(device)
        inv_lazy_A = get_lazy_random_walk(neg_A_hat, I_N, beta=beta_neg)
        inv_features = torch.spmm(inv_lazy_A, pre_features)
        # add for multi-scale version
        temp_sum += (features+inv_features)/2.0
        beta *= args.decline
        beta_neg *= args.decline_neg

    embedds = alpha*X + (1-alpha)*(temp_sum/(K*1.0))
    return embedds


adj_symmetric_normalized, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, 'AugNormAdj', None)
adj_symmetric_normalized = adj_symmetric_normalized.to(device)
features = features.to(device)
labels = labels.to(device)

org_edge_number = my_get_adj_org(args.dataset).count_nonzero()
print(org_edge_number)
org_edge_number = adj_symmetric_normalized.coalesce().indices().size(1) - features.size(0)
print(org_edge_number)
# ------ the inverser random adj -------
negative_rate = 20.0 # negative sampling rate for a negative (i.e., cannot link) graph
neg_avg_edge_num = int((negative_rate*org_edge_number)/(features.shape[0]))
if neg_avg_edge_num % 2 != 0: neg_avg_edge_num += 1 # need n*k odd, for networkx

embedds= GGCM(features, adj_symmetric_normalized, args.alpha, neg_avg_edge_num=neg_avg_edge_num)
loss_val, val_acc, test_acc = eval_classify(embedds, labels, idx_train, idx_val, idx_test, args.epochs, args.lr, args.wdlist)
print('Final test acc:', test_acc)