
import os.path as osp
dataset = 'AmazonProducts'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)

import torch_geometric.transforms as T
from torch_geometric.datasets import AmazonProducts,Yelp
dataset = AmazonProducts(root=path)
print(dataset[0])