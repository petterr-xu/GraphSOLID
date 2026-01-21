import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from torch_geometric.utils import train_test_split_edges,negative_sampling

from src import solid
from dataset import heterosub_dataset
from args import parse_args
from src.utils import VNG_utils, graphbuilder
from solid_trainer import SolidTrainer
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet
warnings.filterwarnings("ignore")


args = parse_args()
print(args)
reweight = False
timestamp_format = "%Y%m%d_%H%M%S"

device = args.device
root_path = osp.dirname(osp.realpath(__file__))
loader = GraphDataLoader()
data_path = osp.join(root_path, 'data', args.dataset, 'data', args.dataset + '.mat')
cnfg_path = osp.join(root_path, 'data', args.dataset, 'meta', args.dataset + '.json')
hetero_ctx = loader.load_from_config(cnfg_path, data_path)
target = hetero_ctx.target_node  # 'review' 或 'user'
data = hetero_ctx.g.to(device)
n_feat = hetero_ctx.n_features
n_cls = hetero_ctx.n_classes
print(data)

repeatition = 5
max_n=500
overall_test_acc, overall_val_acc, overall_val_f1, overall_test_bacc, overall_test_f1 = [], [], [], [], []
overall_mi_recall = []
overall_ma_recall = []
mi_recall = []
ma_recall = []

if args.dataset in ['YelpChi', 'Amazon-Products']:
    data_train_mask, data_val_mask, data_test_mask = data[target].train_mask.clone(), data[target].val_mask.clone(), data[target].test_mask.clone()
    stats = data[target].y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = VNG_utils.get_idx_info(data[target].y, n_cls, data_train_mask)
    class_num_list = n_data
    print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
    class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(data, target, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
    # 更新 HeteroData
    hetero_ctx.g[hetero_ctx.target_node].train_mask = data_train_mask
    # 更新边索引 (可选，取决于是否想物理删除边)
    if not args.keep_edge:
        for etype, mask in edge_mask_dict.items():
            hetero_ctx.g[etype].edge_index = hetero_ctx.g[etype].edge_index[:, mask]
    print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
    minority_mask = class_num_list < (sum(class_num_list)/n_cls)
    minority_class = [i for i in range(n_cls) if minority_mask[i]]
    print("minority classes {}".format(minority_class))
else:
    raise NotImplementedError("Not implemented for dataset {}".format(args.dataset))

# 1. 实例化训练集 (如果是第一次运行，会触发 process 逻辑)
train_dataset = heterosub_dataset.HeteroSubgraphDataset(root='./my_dataset', original_data=data, num_parts=200, split='train')

# 2. 实例化验证集和测试集 (此时 process 不会重复运行，而是直接加载已有的文件)
val_dataset = heterosub_dataset.HeteroSubgraphDataset(root='./my_dataset', split='val')
test_dataset = heterosub_dataset.HeteroSubgraphDataset(root='./my_dataset', split='test')
# 3. 放入 PyG DataLoader
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

for batch in train_loader:
    # batch 现在是一个包含 32 个异构子图的大 Batch
    # 可以直接送入模型：out = model(batch.x_dict, batch.edge_index_dict)
    print(batch)
    break