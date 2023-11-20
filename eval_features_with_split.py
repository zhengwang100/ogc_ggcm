import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from metrics import accuracy
from utils import use_cuda, LinearNeuralNetwork

def eval_classify(X, labels, idx_train, idx_val, idx_test, epochs, lr, wdlist):
    ''' Evaluate embedding by classification with the given split setting
    '''
    device = use_cuda()
    X = X.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    best_acc_val = -1
    best_model = None
    for wd in wdlist:
        model = _train_simple_mlp(X, labels, idx_train, idx_val, idx_test, epochs, lr, wd)
        loss_val, acc_val, acc_test = _validate(model, X, labels, idx_val, idx_test)
        #print('wd', wd, 'loss_val', loss_val, 'acc_val', acc_val, 'Test ACC', acc_test)     
        if(acc_val > best_acc_val):
            best_acc_val = acc_val
            best_model = copy.deepcopy(model)
    loss_val, val_acc, test_acc = _validate(best_model, X, labels, idx_val, idx_test)
    print('val_loss', loss_val, 'val_acc', val_acc, 'test_acc', test_acc)
    return loss_val, val_acc, test_acc

def _validate(model, X, labels, idx_val, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(X)
        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_val.item(), acc_val.item(), acc_test.item()

def _train_simple_mlp(X, labels, idx_train, idx_val, idx_test, epochs, lr, wd):
    device = use_cuda()
    best_model = None
    best_acc_val = -1.0
    model = LinearNeuralNetwork(X.size(1), labels.max().item()+1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val, acc_val, acc_test = _validate(model, X, labels, idx_val, idx_test)
        if(acc_val >= best_acc_val):
            best_acc_val = acc_val
            best_model = copy.deepcopy(model)
        print(epoch, loss_val, acc_val, 'acc_test', acc_test)
        #print(epoch, acc_val, acc_test)
        
    return best_model