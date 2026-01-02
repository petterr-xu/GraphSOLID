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
from src.utils import VNG_utils,tab_dataset_util, graphbuilder
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet
import src.utils.graphbuilder

args = parse_args()
device = 'cuda:0'
dataset = 'YelpChi'
root_path = osp.dirname(osp.realpath(__file__))
data_path = osp.join(root_path, 'data', dataset, 'data', dataset + '.mat')
cnfg_path = osp.join(root_path, 'data', dataset, 'meta', dataset + '.json')


# 只需要两行代码即可完成所有初始化
loader = GraphDataLoader()
ctx = loader.load_from_config(cnfg_path, data_path, device)
metadata = ctx.g.metadata()
n_cls = ctx.n_classes
n_feat = ctx.n_features
data = ctx.g
target = ctx.target_node

data_train_mask, data_val_mask, data_test_mask = data[target].train_mask.clone(), data[target].val_mask.clone(), data[target].test_mask.clone()
stats = data[target].y[data_train_mask]
n_data = []
for i in range(n_cls):
    data_num = (stats == i).sum()
    n_data.append(int(data_num.item()))
idx_info = VNG_utils.get_idx_info(data[target].y, n_cls, data_train_mask)
print("num of class in original training data: {} -> {}".format(n_data,sum(data_train_mask).item()))
class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(data, target, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n=500)
# 更新 HeteroData
ctx.g[target].train_mask = data_train_mask
# 更新边索引 (可选，取决于是否想物理删除边)
if not args.keep_edge:
    for etype, mask in edge_mask_dict.items():
        ctx.g[etype].edge_index = ctx.g[etype].edge_index[:, mask]
print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
minority_mask = class_num_list < (sum(class_num_list)/n_cls)
minority_class = [i for i in range(n_cls) if minority_mask[i]]
print("minority classes {}".format(minority_class))

denoise_nhid = n_feat
denoise_kwargs = {
    "feature_length": denoise_nhid,
    "n_length": args.n_length,
    "n_channels": args.n_channels,
    "ch_mults": args.ch_mults,
    "is_attn": args.is_attn,
    "n_blocks": args.n_blocks,
    "class_channels": args.class_embedding_channel,
    "time_channels": args.time_embedding_channel,
    "num_class": n_cls,
}
eps_model = unet.UNet(**denoise_kwargs)
# eps_model = unet_vector.UNet(denoise_config)
if args.beta_schedule == "lin":
    beta = torch.linspace(args.beta_bound[0], args.beta_bound[1], args.T)
elif args.beta_schedule == "exp":
    beta_exp = args.beta_bound[0] * (args.beta_bound[1] / args.beta_bound[0]) ** (np.arange(args.T) / args.T)
    beta = torch.tensor(beta_exp,dtype=torch.float32)
elif args.beta_schedule == "quad":
    beta_quad = args.beta_bound[0] + (np.arange(args.T) / args.T) ** 2 * (args.beta_bound[1] - args.beta_bound[0])
    beta = torch.tensor(beta_quad,dtype=torch.float32)
else:
    print("NO SUCH BETA SCHEDULE:"+args.beta_schedule)
    raise Exception
teacher_model = teacher.MLPTeacher(n_feat,n_cls,layers=1,drop=0.4).to(device)
diffusion_model = diffusion.GDDPMblock(eps_model,beta,n_steps=args.T,device=device).to(device)
dif_optimizer = torch.optim.Adam(diffusion_model.eps_model.parameters(), lr=args.dif_lr)
# definition of edge learner
node_types = ctx.g.node_types
edge_types = ctx.g.edge_types
# 自动生成维度字典 (根据 ctx.g 的特征形状)
node_dim_dict = {
    node_type: n_feat
    for node_type in node_types
}
edge_decoder = edge_learner.HeteroEdgePredicter(
    node_types=node_types,
    edge_types=edge_types,
    node_dim_dict=node_dim_dict,
    n_hid=args.decoder_hid
).to(device)
v_information, src_idx = solid.softlabel_based_hard_nodes_sampling(data[ctx.target_node].x[data[ctx.target_node].train_mask],
                                                    data[ctx.target_node].y[data[ctx.target_node].train_mask],
                                                    n_cls,
                                                    diffusion_model = diffusion_model,
                                                    teacher = teacher_model,
                                                    args = args,
                                                    device=device)
# construct new nodes and edges then augment the graph
new_node_num = v_information['feat'].shape[0]
print("{} new nodes".format(new_node_num))

aug_data = solid.add_new_hetero_nodes_all_relations(data,
                                                    v_information['feat'],
                                                    v_information['label'],
                                                    edge_decoder,
                                                    target_node=ctx.target_node,
                                                    device=device)