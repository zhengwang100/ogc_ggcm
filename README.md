# OGC and GGCM - PyTorch Implementation

This repository contains the original PyTorch implementation of OGC and GGCM for semi-supervised node classification [1].

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-cluster-assumption-to-graph-convolution/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=from-cluster-assumption-to-graph-convolution)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-cluster-assumption-to-graph-convolution/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=from-cluster-assumption-to-graph-convolution)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-cluster-assumption-to-graph-convolution/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=from-cluster-assumption-to-graph-convolution)

## ❗ News 

### PyG Implementation:
- OGC (official repository): [pytorch_geometric/examples/ogc.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogc.py)
- GGCM: [pytorch_geometric/examples/ggcm.py](https://github.com/xYix/pytorch_geometric/blob/master/examples/ggcm.py)


### DGL Implementation:
- OGC (official repository): [dgl/examples/pytorch/ogc](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogc)
- GGCM: [dgl/examples/pytorch/ggcm](https://github.com/SinuoXu/dgl/tree/master/examples/pytorch/ggcm)

### rLLM Implementation:
- OGC (official repository): [rllm/examples/ogc.py](https://github.com/rllm-team/rllm/blob/main/examples/ogc.py)


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

