# OGC and GGCM - PyTorch Implementation

This repository contains the original PyTorch implementation of OGC and GGCM for semi-supervised node classification [1].

## ❗ News 

### PyG Implementation:
- OGC (official repository): [pytorch_geometric/examples/ogc.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogc.py)
- GGCM: [pytorch_geometric/examples/ggcm.py](https://github.com/xYix/pytorch_geometric/blob/master/examples/ggcm.py)


### DGL Implementation:
- OGC (official repository): [dgl/examples/pytorch/ogc](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogc)
- GGCM: Coming soon~


## Original Codes

### Running OGC:
``` 
python main_ogc.py --dataset cora 
python main_ogc.py --dataset citeseer 
python main_ogc.py --dataset pubmed 
```

### Running GGCM:
``` 
python main_ggcm.py --dataset cora
python main_ggcm.py --dataset citeseer
python main_ggcm.py --dataset pubmed
``` 
## References

[1] Wang, Zheng, Hongming Ding, Li Pan, Jianhua Li, Zhiguo Gong, and Philip S. Yu. "From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited." arXiv preprint arXiv:2309.13599 (2023).
