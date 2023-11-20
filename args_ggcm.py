import argparse

def get_citation_args(dataset='cora'):
    if(dataset=='cora'): 
        return _get_citation_args_cora()
    if(dataset=='citeseer'): 
        return _get_citation_args_citeseer()
    if(dataset=='pubmed'):
        return _get_citation_args_pubmed()

def _get_citation_args_cora():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora", help='Dataset to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
    parser.add_argument('--wdlist', type=float, nargs='*', default=[1e-5], help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha.')
    parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')
    parser.add_argument('--decline', type=float, default=1.0, help='decline.')
    parser.add_argument('--decline_neg', type=float, default=0.5, help='decline negative.')
    args, _ = parser.parse_known_args()
    return args

def _get_citation_args_citeseer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="citeseer", help='Dataset to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
    parser.add_argument('--wdlist', type=float, nargs='*', default=[1e-3], help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha.')
    parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')
    parser.add_argument('--decline', type=float, default=1.0, help='decline.')
    parser.add_argument('--decline_neg', type=float, default=1.0, help='decline negative.')
    args, _ = parser.parse_known_args()
    return args

def _get_citation_args_pubmed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="pubmed", help='Dataset to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
    parser.add_argument('--wdlist', type=float, nargs='*', default=[2e-5], help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha.')
    parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')
    parser.add_argument('--decline', type=float, default=1.0, help='decline.')
    parser.add_argument('--decline_neg', type=float, default=0.5, help='decline negative.')
    args, _ = parser.parse_known_args()
    return args