import torch.nn.functional as F
import torch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def check_preds_change(preds1, preds2):
    labels = preds2.max(1)[1]
    return accuracy(preds1, labels).item()

def validate(model, X, labels_one_hot, labels, idx_train_val, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(X)
        loss_train_val = F.mse_loss(output[idx_train_val], labels_one_hot[idx_train_val])
        acc_train_val = accuracy(output[idx_train_val], labels[idx_train_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_train_val.item(), acc_train_val.item(), acc_test.item(), output[idx_test]