import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges,negative_sampling

from solid_trainer import SolidTrainer
from args import parse_args
from src import solid
from src.utils import VNG_utils,tab_dataset_util
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet
import src.utils.graphbuilder

device = 'cuda:0'
dataset = 'YelpChi'
root_path = osp.dirname(osp.realpath(__file__))
data_path = osp.join(root_path, 'data', dataset, 'data', dataset + '.mat')
cnfg_path = osp.join(root_path, 'data', dataset, 'meta', dataset + '.json')


# 只需要两行代码即可完成所有初始化
loader = GraphDataLoader()
ctx = loader.load_from_config(cnfg_path, data_path, device)
metadata = ctx.g.metadata()
print(ctx.g.node_types)
for edge_type in ctx.g.edge_types:
    print(ctx.g[edge_type].val_pos_edge_index.shape)

print(metadata)
print(ctx.target_node)
model = HeteroNN.HeteroGNN_classifier(ctx.target_node, ctx.g.metadata(), nhid=32, nclass=ctx.n_classes, nlayer=2, dropout=0.5)
model = model.to(device)

# 1. 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 2. 准备训练数据
target = ctx.target_node
x_dict = ctx.g.x_dict
edge_index_dict = ctx.g.edge_index_dict
y = ctx.g[target].y
train_mask = ctx.g[target].train_mask

# 3. 训练循环
print(f"开始训练数据集: {dataset}")
model.train()
for epoch in range(1001):
    optimizer.zero_grad()
    
    # 前向传播：得到的是所有节点类型的 Embedding 字典
    out_dict = model(x_dict, edge_index_dict)
    
    # 取出目标节点的预测结果 [num_nodes, hidden_channels]
    out = out_dict
    
    # 计算损失 (PyG 的 CrossEntropy 允许输入未经过 Softmax 的特征)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# 4. 简单测试
model.eval()
with torch.no_grad():
    out_dict = model(x_dict, edge_index_dict)
    pred = out_dict.argmax(dim=1)
    acc = (pred[ctx.g[target].test_mask] == y[ctx.g[target].test_mask]).sum() / ctx.g[target].test_mask.sum()
    print(f"测试集准确率: {acc:.4f}")
