import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sparse_mx_to_torch_sparse_tensor, LinearNeuralNetwork
from metrics import check_preds_change, validate
from ogc_utils import build_label_indicator_matrix, get_one_hot_labels

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="citeseer", choices=["cora", "citeseer", "pubmed"], help='Dataset to use.')
args, _ = parser.parse_known_args()

decline = 0.9           # the decline rate 
eta_sup = 0.001         # the learning rate for supervised loss
eta_W = 0.5             # the learning rate for updating W
beta = 0.1              # in [0,1], the moving probability that a node moves to its neighbors 
max_similar_tol = 0.995 # the max_tol test set label prediction similarity between two iterations 
max_tolerance = 2       # the tolreance for consecutively getting very similar test prediction

def get_opt_gc_features(U, adj, Y, predY, W, S, I_N):
    """
       Get optimized graph convolution features, by upadate embedding X based on : 
           1) graph structure via LGC and 2) supervised labels via SEB
           i.e., X = LGC(U) + SEB(U)
       Args:
           U: the learned node embedding matrix with shape [n, d]
           adj: adjacency matrix of the graph with shape (n, n) 
           Y: given labels with shape [n_label, c] 
           predY: the predicted labels with shape [n_label, c]
           W: learnable weight matrix with shape [c, d]
           S: label indicator matrix wiht shape [n, n]
        Return U[n,d] the final embedding results
    """
    global eta_sup, beta, decline
    U = torch.FloatTensor(U)
    
    # ------ update the smoothness loss via LGC ------
    lazy_A = beta*adj + (1-beta)*I_N
    U = torch.spmm(lazy_A, U) # H = [beta*A_hat +(1-beta)*I]H
    
    # ------ update the supervised loss via SEB ------
    dU_sup = 2*np.linalg.multi_dot([S, (-Y+predY), W]) # note: in pytorch, W[c,d]
    U = U - eta_sup*torch.FloatTensor(dU_sup)
    
    eta_sup = eta_sup*decline
    return U

def train_one_layer_fc_model(U):
    # For updating W, train a global linear model (like Y=WX) one epoch
    optimizer = optim.SGD(linear_clf.parameters(), lr=eta_W)
    for _ in range(1):
        linear_clf.train()
        optimizer.zero_grad()
        output = linear_clf(U)
        loss_train_val = F.mse_loss(output[idx_train_val], labels_one_hot[idx_train_val], reduction='sum')
        loss_train_val.backward()
        optimizer.step()
    
def OGC(U, adj, S, I_N):
    """
    Optimized Graph Convolution (OGC).
    OGC alternatively updates W and U at each iteration, as follows:
        --- update W by training a simple linear classifier Y=WX.
        --- update U by LGC(U) + SEB(U).
    Args:
           U: the learned node embedding matrix with shape [n, d]
           adj: adjacency matrix of the graph with shape (n, n) 
           S: label indicator matrix wiht shape [n, n]
    Return: classification acc on test set
    """
    patience = 0
    _, _, last_acc_test, last_test_preds = validate(linear_clf, U, labels_one_hot, labels, idx_train_val, idx_test)
    for iternum in range(64):
        # updating W by training a simple linear supervised model Y=W*X
        train_one_layer_fc_model(U)
        # updating U by LGC and SEB jointly
        U = get_opt_gc_features(U.numpy(), adj, labels_one_hot.numpy(), linear_clf(U).data.numpy(), linear_clf.W.weight.data.numpy(), S, I_N)
        
        # show and check 
        loss_train_val, acc_train_val, acc_test, test_preds = validate(linear_clf, U, labels_one_hot, labels, idx_train_val, idx_test)
        print(iternum+1, 'loss_train_val', loss_train_val, 'acc_train_val', acc_train_val, 'acc_test', acc_test)
        similar_rate = check_preds_change(last_test_preds, test_preds)

        if(similar_rate > max_similar_tol):
            patience += 1
            if(patience > max_tolerance):
                break
        last_test_preds = test_preds
        last_acc_test = acc_test
        
    return last_acc_test


adj_symmetric_normalized, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, 'AugNormAdj', None)
labels_one_hot = get_one_hot_labels(labels)
I_N = sparse_mx_to_torch_sparse_tensor(sp.eye(features.shape[0])).float()
idx_train_val = torch.cat((idx_train, idx_val), 0) # use both train and validation sets for model learning

label_indicator_matrix = build_label_indicator_matrix(labels.shape[0], idx_train) # less is more (LIM) trick
#label_indicator_matrix = build_label_indicator_matrix(labels.shape[0], idx_train_val) # the traditional way

linear_clf = LinearNeuralNetwork(nfeat=features.size(1), nclass=labels.max().item()+1, bias=False) # a linear supervised model (Y=WX) for learning W
res = OGC(features, adj_symmetric_normalized, label_indicator_matrix, I_N)
print('Test Acc:', res)