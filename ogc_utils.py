import numpy as np
import torch

def get_one_hot_labels(labels):
    nb_classes=labels.max().item()+1
    labels_one_hot = torch.zeros(len(labels), nb_classes)
    labels_one_hot[torch.arange(len(labels)), labels] = 1
    return labels_one_hot

def build_label_indicator_matrix(n, idx_train):
    """ C - [n,n] diag matrix with ii=1 if node i is labeled """
    C = np.zeros([n, n])
    for node in idx_train:
        C[int(node), int(node)] = 1.0
    return C