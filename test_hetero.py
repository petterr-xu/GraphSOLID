import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from torch_geometric.utils import train_test_split_edges,negative_sampling

from solid_trainer import SolidTrainer
from args import parse_args
from src import solid
from src.utils import VNG_utils,tab_dataset_util
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.denoise import unet
import src.utils.graphbuilder

device = 'cpu'
dataset = 'YelpChi'
root_path = osp.dirname(osp.realpath(__file__))
data_path = osp.join(root_path, 'data', dataset, 'data', dataset + '.mat')
cnfg_path = osp.join(root_path, 'data', dataset, 'meta', dataset + '.json')


# 只需要两行代码即可完成所有初始化
loader = GraphDataLoader()
ctx = loader.load_from_config(cnfg_path, data_path)

# 训练时直接引用 ctx.g
# model = HeteroRGCN(ctx.g.metadata(), ctx.input_dim, ctx.num_classes)


