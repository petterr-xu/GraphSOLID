import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import os
import re
import csv
import copy
import random
import numpy as np
import pandas as pd
import os.path as osp
import networkx as nx
import scipy.io as sio
from scipy import linalg
import scipy.sparse as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from torch_geometric.data import HeteroData
from matplotlib.animation import FuncAnimation

from sklearn.metrics import classification_report,roc_auc_score

def load(model:nn.Module, path, device="cuda:0"):
    ckpt = torch.load(path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        return
    model.load_state_dict(torch.load(path))

def load_from_class(model_class, file_path, device="cuda:0"):
    ckpt = torch.load(file_path, map_location=device)
    config = ckpt["config"]

    # 利用 config 重建模型
    model = model_class(**config)
    model.load_state_dict(ckpt["state_dict"])
    return model

def softmax_with_temperature(logits, temperature):
    adjusted_logits = logits / temperature
    softmax_output = F.softmax(adjusted_logits, dim=-1)
    return softmax_output

def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1), device=X.device)
    rows = torch.zeros(X.shape[0] * (k + 1), device=X.device)
    cols = torch.zeros(X.shape[0] * (k + 1), device=X.device)   
    norm_row = torch.zeros(X.shape[0], device=X.device)
    norm_col = torch.zeros(X.shape[0], device=X.device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label)).to(label.device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info
    
def get_dataset(name, path, split_type='public',normalize_features=False):
    import torch_geometric.transforms as T
    from torch_geometric.transforms import RandomNodeSplit
    transform=T.NormalizeFeatures() if normalize_features else None
    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        return Planetoid(path, name, transform=transform, split=split_type)
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='computers', transform=transform)
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='photo', transform=transform)
    elif name == 'Coauthor-CS':
        from torch_geometric.datasets import Coauthor
        return Coauthor(root=path, name='cs', transform=transform)
    elif name == 'Amazon-Products':
        data = load_amazon_hetero(path)
        transform = RandomNodeSplit(num_val=0.2, num_test=0.4)
        data = transform(data)
        return data
    elif name == 'YelpChi':
        data = load_yelp_hetero(path)
        transform = RandomNodeSplit(num_val=0.2, num_test=0.4)
        data = transform(data)
        return data
    else:
        raise NotImplementedError("Not Implemented Dataset!")
    
def load_yelp_hetero(file_path):
    """
    加载 YelpChi 异构欺诈数据集
    节点: Review (评论)
    关系: 
        - rur: 相同用户发布的评论
        - rsr: 对相同产品且评分相同的评论
        - rtr: 同一月内对相同产品的评论
    """
    file_path = osp.join(file_path, 'YelpChi.mat')
    mat = sio.loadmat(file_path)
    data = HeteroData()
    
    # 处理节点特征与标签
    # .todense() 确保稀疏矩阵转为稠密张量
    x = torch.from_numpy(mat['features'].todense()).float()
    y = torch.from_numpy(mat['label'].flatten()).long()
    
    data['review'].x = x
    data['review'].y = y
    
    # 处理三种异构边关系
    # mat['net_rur'] 等通常是 scipy sparse 矩阵
    for rel_type in ['rur', 'rsr', 'rtr']:
        adj = mat[f'net_{rel_type}']
        row, col = adj.nonzero()
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        # 定义异构边：(源节点, 关系名, 目标节点)
        data['review', rel_type, 'review'].edge_index = edge_index
        
    print(f"YelpChi load success: nodes {x.shape[0]}, edge type: rur, rsr, rtr")
    return data

def load_amazon_hetero(file_path):
    """
    加载 Amazon 异构欺诈数据集
    节点: User (用户)
    关系:
        - upu: 购买过至少一个相同产品的用户
        - usu: 一周内给出过相同评分的用户
        - uvu: 评论文本相似度最高的前 5% 的用户对
    """
    file_path = osp.join(file_path, 'Amazon.mat')
    mat = sio.loadmat(file_path)
    data = HeteroData()
    
    x = torch.from_numpy(mat['features'].todense()).float()
    y = torch.from_numpy(mat['label'].flatten()).long()
    
    data['user'].x = x
    data['user'].y = y
    
    # 处理三种异构边关系
    for rel_type in ['upu', 'usu', 'uvu']:
        adj = mat[f'net_{rel_type}']
        row, col = adj.nonzero()
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        data['user', rel_type, 'user'].edge_index = edge_index
        
    print(f"Amazon load success: nodes {x.shape[0]}, edge type: upu, usu, uvu")
    return data

## Construct random removal ##
def make_random_data_remove(edge_index, label, n_data, n_cls, train_num, train_mask):

    assert len(train_num) == n_cls, "length of train_num must be consistent with n_cls"
    train_num = np.array(train_num)
    for i in range(n_cls):
        assert train_num[i] <= n_data[i], f"number of class {i} must more than reserved number"

    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask),device=train_mask.device)
    original_mask = train_mask.clone() 

    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in range(n_cls):
        cls_idx = cls_idx_list[i]
        if len(cls_idx) > train_num[i]:
            perm = torch.randperm(len(cls_idx))
            keep_idx = cls_idx[perm[:train_num[i]]]
            remove_idx = cls_idx[perm[train_num[i]:]]
            remove_idx_list[i] = list(remove_idx.to('cpu').numpy())

    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(train_num), train_mask, idx_info, node_mask, edge_mask


def get_step_split(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    base_valid_each = valid_each

    head_list = [i for i in range(nclass//2)] 

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node

def confidence_dis(soft_labels:torch.Tensor,hard_labels:torch.Tensor,num_classes):
    if hard_labels.dim() > 1:
        hard_labels = hard_labels.argmax(1)
    dis = []
    for class_ in range(num_classes):
        label_mask = (hard_labels == class_)
        class_soft_label = soft_labels[label_mask]
        dis.append(torch.mean(class_soft_label,dim=0))
    dis = torch.stack(dis)
    return dis


def class_dis(labels,num_classes=None,norm=False):
    if num_classes == None:
        num_classes = torch.max(labels)-torch.min(labels)+1
    class_dis = torch.empty([num_classes],dtype=torch.int32)
    for node_class in range(num_classes):
        class_num = torch.sum((labels == node_class).view(-1),dtype=torch.int32)
        class_dis[node_class] = class_num
    if norm :
        class_dis = class_dis / torch.sum(class_dis,dtype=torch.float32)
    return class_dis


# def imbalanced_train_schedule(graph:dgl.DGLGraph,imb_ratio,num_class,val_size,test_size,bias=0,minimum_train_size=None):
#     assert imb_ratio > 1, "Imbalanced ratio must higher then 1, but get {}.".format(imb_ratio)
#     class_dis = node_class_dis(graph,num_classes=num_class)
#     class_size_rank,class_rank = torch.sort(class_dis,descending=True)
#     majority_train_size = class_size_rank[0]-val_size-test_size-bias
#     mu = np.power(1/imb_ratio, 1/(num_class - 1))
#     train_schedule = torch.empty([num_class],dtype=torch.int32)
#     for i in range(num_class):
#         class_ = class_rank[i]
#         train_schedule[class_] = int(majority_train_size * np.power(mu, i))
#         assert (train_schedule[class_] + val_size + test_size) <= class_size_rank[i],"Schedule out of maximum size for train {}, val {}, test {} while class {} only has {} nodes".format(
#             train_schedule[class_],val_size,test_size,class_,class_size_rank[i])
#         if minimum_train_size is not None:
#             assert train_schedule[class_] >= minimum_train_size,"Train size must higher than {}, however class {} gets {} train size while imbalance ratio is {}".format(
#                 minimum_train_size,class_,train_schedule[class_],imb_ratio)
#     return train_schedule

# def edge_dataset_split(graph:dgl.DGLGraph,val_test_ratio=1):
#     node_id = torch.tensor([i for i in range(graph.num_nodes()) if graph.ndata['train_mask'][i]]).to(graph.device)
#     sub_graph = dgl.out_subgraph(graph,node_id)
#     train_pos_u, train_pos_v = sub_graph.edges()
#     u, v = graph.edges()

#     # 将子图边列表中的边存储在一个集合中
#     train_edges_set = set(zip(train_pos_u.tolist(), train_pos_v.tolist()))

#     # 初始化新的边列表，用于存储删除子图边后的边
#     rest_u = []
#     rest_v = []

#     # 遍历第一个边列表，检查每一条边是否在子图边列表中
#     for src, dst in zip(u.tolist(), v.tolist()):
#         if (src, dst) not in train_edges_set:
#             rest_u.append(src)
#             rest_v.append(dst)
#     rest_u = torch.tensor(rest_u,device=graph.device)
#     rest_v = torch.tensor(rest_v,device=graph.device)
#     num_rest_edges = len(rest_u)
#     num_val = (val_test_ratio * num_rest_edges) // (val_test_ratio + 1)
#     eids = np.arange(num_rest_edges)
#     eids = np.random.permutation(eids)
#     val_pos_u, val_pos_v = rest_u[eids[0:num_val]], rest_v[eids[0:num_val]]
#     test_pos_u, test_pos_v = rest_u[eids[num_val:]], rest_v[eids[num_val:]]

#     adj = sp.coo_matrix((np.ones(len(u.cpu())), (u.cpu().numpy(), v.cpu().numpy())))
#     # 2708*2708 空边
#     adj_neg = 1-adj.todense()-np.eye(graph.num_nodes())
#     neg_u, neg_v = np.where(adj_neg!=0)
#     neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())
#     start = 0
#     end = len(train_pos_u)
#     train_neg_u, train_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
#     start = end
#     end = start+len(val_pos_u)
#     val_neg_u, val_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
#     start = end
#     end = start+len(test_pos_u)
#     test_neg_u, test_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
#     #positive graph
#     train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
#     val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())
#     test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
#     #negative graph
#     train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
#     val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())
#     test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())
#     return [train_pos_g,val_pos_g,test_pos_g],[train_neg_g,val_neg_g,test_neg_g]


def random_edge_dataset_split(graph:dgl.DGLGraph,split_schedule:dict):
    try:
        train_schedule = split_schedule["train"]
        val_schedule = split_schedule["val"]
        test_schedule = split_schedule["test"]
    except Exception as e:
        print("split_schedule must contain \"train\", \"val\", \"test\"")
    u, v = graph.edges()
    eids = np.arange(graph.num_edges())
    eids = np.random.permutation(eids)
    #positive examples
    start = 0
    end = train_schedule
    train_pos_u, train_pos_v = u[eids[start:end]], v[eids[start:end]]
    start = end
    end = start+val_schedule
    val_pos_u, val_pos_v = u[eids[start:end]], v[eids[start:end]]
    start = end
    end = start+test_schedule
    test_pos_u, test_pos_v = u[eids[start:end]], v[eids[start:end]]
    adj = sp.coo_matrix((np.ones(len(u.cpu())), (u.cpu().numpy(), v.cpu().numpy())))
    # 2708*2708 空边
    adj_neg = 1-adj.todense()-np.eye(graph.num_nodes())
    neg_u, neg_v = np.where(adj_neg!=0)
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())
    start = 0
    end = train_schedule
    train_neg_u, train_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
    start = end
    end = start+val_schedule
    val_neg_u, val_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
    start = end
    end = start+test_schedule
    test_neg_u, test_neg_v = neg_u[neg_eids[start:end]], neg_v[neg_eids[start:end]]
    # train_g = dgl.remove_edges(graph, eids[train_schedule:train_schedule+val_schedule+test_schedule])
    #positive graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    #negative graph
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())
    return [train_pos_g,val_pos_g,test_pos_g],[train_neg_g,val_neg_g,test_neg_g]

def _graph_dataset_split(graph:dgl.DGLGraph,num_classes,split_schedule:dict):
    try:
        train_schedule = split_schedule["train"]
        val_schedule = split_schedule["val"]
    except Exception as e:
        print("split_schedule must contain \"train\", \"val\" ")
    assert (len(train_schedule) == len(val_schedule) == num_classes), "schedule must contain all node class."
    ori_train_mask = graph.ndata['train_mask']
    ori_val_mask = graph.ndata['val_mask']
    # ori_test_mask = graph.ndata['test_mask']
    num_nodes = graph.num_nodes()
    device = graph.device
    assert (sum(train_schedule)+sum(val_schedule) <= num_nodes), "required nodes are beyond the dataset."
    train_mask = torch.full((num_nodes,),False,dtype=torch.bool)
    val_mask = torch.full((num_nodes,),False,dtype=torch.bool)
    print("split schedule:",split_schedule)
    train_num = [sum(graph.ndata['label'][ori_train_mask].argmax(1) == label) for label in range(num_classes)]
    print("oringinal train num",train_num)
    val_num = [sum(graph.ndata['label'][ori_val_mask].argmax(1) == label) for label in range(num_classes)]
    print("oringinal val num",val_num)
    for label in range(num_classes):
        assert (train_schedule[label] > 0 and val_schedule[label] > 0) , "number of required nodes must greater then 0."
        label_indices = torch.tensor([i for i in range(num_nodes) if ori_train_mask[i] and graph.ndata['label'][i].argmax(0) == label])
        assert len(label_indices) >= train_schedule[label] , "only {} nodes of label {} in train dataset, however require {} nodes.".format(len(label_indices),label,train_schedule[label])
        rand_indices = random.sample(range(len(label_indices)),train_schedule[label])
        train_mask[label_indices[torch.tensor(rand_indices)]] = True

        label_indices = torch.tensor([i for i in range(num_nodes)if ori_val_mask[i] and graph.ndata['label'][i].argmax(0) == label])
        assert len(label_indices) >= val_schedule[label] , "only {} nodes of label {} in evaluation dataset, however require {} nodes.".format(len(label_indices),label,val_schedule[label])
        rand_indices = random.sample(range(len(label_indices)),val_schedule[label])
        val_mask[label_indices[torch.tensor(rand_indices)]] = True
    train_indices = [i for i in range(len(train_mask)) if train_mask[i]]
    val_indices = [i for i in range(len(val_mask)) if val_mask[i]]
    is_overlap = any(element in train_indices for element in val_indices)
    assert ~is_overlap,"there is overlap between valmask and train mask"
    graph.ndata["train_mask"] = train_mask.to(device)
    graph.ndata["val_mask"] = val_mask.to(device)
    return graph

def dis_based_class_mask(labels: torch.Tensor, class_dis, num_class, mask_size, beta, adjustment_factor=1.0, device="cuda:0"):
    """
    Args:
        labels (torch.Tensor): 存储不同样本的label值
        class_dis (torch.Tensor): 归一化的标签数量分布 (即每个类别的样本比例)
        num_class (int): 类别总数
        mask_size (int): 掩码的大小，即要生成的掩码的总数量
        beta (float): 掩码中True的比例，即样本总体的掩码概率
        adjustment_factor (float): 调整因子，用于控制掩码调整的力度
        device (str): 设备（默认cuda:0）
        
    Returns:
        mask (torch.Tensor): 生成的掩码
    """
    if labels.dim() > 1:
        labels = labels.argmax(1)
    # 计算每个类别在掩码中True的数量概率，应用调整因子
    adjusted_probs = (class_dis ** adjustment_factor)
    adjusted_probs = adjusted_probs / adjusted_probs.sum()  # 重新归一化调整后的概率
    
    # 生成掩码
    mask = torch.zeros(mask_size, device=device, dtype=torch.bool)
    
    # 获取每个类别的掩码数量
    class_indices = [torch.where(labels == i)[0] for i in range(num_class)]
    
    for i, indices in enumerate(class_indices):
        prob = adjusted_probs[i] * beta / class_dis[i]   # 为每个类别计算出概率
        assert prob < 1., "prob must lower than 1.0, consider higher adjustment factor."
        if len(indices) > 0:
            # 使用伯努利分布随机选择掩码中的True数量
            selected_mask = torch.bernoulli(torch.full((indices.size(0),), prob, device=device)).bool()
            mask[indices[selected_mask]] = True
    
    return mask

# def load_cora_aug(load_cache_file:bool = True, file_path = "CGDM-Im\\dataset\\aug_graph\\CoraAug_gs0.bin"):
#     file_path = file_path.replace("\\",os.sep)
#     dgl_graph_list,_ = dgl.load_graphs(file_path)
#     return dgl_graph_list[0]

# def load_cora_emb(load_cache_file:bool = True, file_path = "CGDM-Im\\dataset\\emb_graph\\Cora_emb.bin"):
#     file_path = file_path.replace("\\",os.sep)
#     dgl_graph_list,_ = dgl.load_graphs(file_path)
#     return dgl_graph_list[0]

def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def gather_image(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def gather_vector(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1)

# FID
def fid(real_features, gen_features):
    # 计算真实样本和生成样本的均值和协方差矩阵
    mu_real = np.mean(real_features.detach().to('cpu').numpy(), axis=0)
    sigma_real = np.cov(real_features.detach().to('cpu').numpy(), rowvar=False)

    mu_gen = np.mean(gen_features.detach().to('cpu').numpy(), axis=0)
    sigma_gen = np.cov(gen_features.detach().to('cpu').numpy(), rowvar=False)

    # 计算均值差异的平方
    diff = mu_real - mu_gen
    diff_squared = np.sum(diff ** 2)

    # 计算协方差矩阵的乘积和平方根
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    # 防止计算过程中出现复数值
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 FID
    fid = diff_squared + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def show_sample_dis(x_o, x_g, y_o, y_g, num_class, title, path , save=True):
    norm = Normalize(vmin=0, vmax=num_class-1)
    colors = cm.get_cmap('tab10', num_class)
    # 可视化结果
    plt.figure(figsize=(10, 8))
    # 可视化原始样本，使用圆形点（默认形状 'o'）
    scatter_original = plt.scatter(x_o[:,0], x_o[:, 1], 
                                    c=y_o, cmap=colors, norm=norm, marker='o', label="real sapmles",
                                    alpha=0.6, edgecolor='none')
    # 可视化生成样本，使用三角形点（形状 '^'），并添加黑色边框
    scatter_generated = plt.scatter(x_g[:,0], x_g[:, 1], 
                                    c=y_g, cmap=colors, norm=norm, marker='^', 
                                    label="synthesized sapmles", alpha=0.6, edgecolor='k', linewidth=0.8)
    
    # 添加颜色条以显示类别映射
    cbar = plt.colorbar(scatter_original)
    cbar.set_label('Class')
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig(path.replace("\\",os.sep))
    plt.close()


def show_samples_dis(data, fold, n_generated_nodes, num_class = 6, trans=None):
    embedding = data.x.detach().cpu().numpy()
    # if trans is not None:
    #     embedding = trans(embedding)
    generate_mask = np.array([0] * (len(embedding) - n_generated_nodes) + [1] * n_generated_nodes)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embedding)
    y_original_np = data.y[:-n_generated_nodes].cpu().detach().numpy()
    y_generated_np = data.y[-n_generated_nodes:].cpu().detach().numpy()
    # print(y_generated_np)
    # 所有样本
    show_sample_dis(X_tsne[generate_mask == 0],
                    X_tsne[generate_mask == 1],
                    y_original_np,
                    y_generated_np,
                    num_class,
                    "All Samples",
                    fold + "\\all_samples.svg")
    # 训练集和生成样本
    show_sample_dis(X_tsne[np.logical_and(generate_mask == 0, data.train_mask.cpu().detach().numpy())],
                    X_tsne[generate_mask == 1],
                    data.y[data.train_mask][:-n_generated_nodes].cpu().detach().numpy(),
                    y_generated_np,
                    num_class,
                    "Train Samples",
                    fold + "\\train_samples.svg")

    # 测试集和生成样本
    show_sample_dis(X_tsne[np.logical_and(generate_mask == 0, data.test_mask.cpu().detach().numpy())],
                    X_tsne[generate_mask == 1],
                    data.y[data.test_mask].cpu().detach().numpy(),
                    y_generated_np,
                    num_class,
                    "Test Samples",
                    fold + "\\test_samples.svg")


def show_embedding_dis(embedding:torch.Tensor,labels:torch.Tensor,num_classes:int):
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(embedding.cpu())
    colors=['b', 'c', 'y', 'm', 'r', 'g', 'k','yellow','yellowgreen','wheat']
    assert (num_classes <= len(colors)) , "could not assign each class to a specific color"
    for class_ in range(num_classes):
        mask = (labels == class_).cpu()
        vis_x = x_tsne[:,0]
        vis_y = x_tsne[:,1]
        plt.scatter(vis_x[mask], vis_y[mask], c=colors[class_], marker='h',label=str(class_))
    plt.legend()
    plt.savefig(".\\history_data\\figure\\embedding_dis.png".replace("\\",os.sep))
    # plt.show()
    plt.close()

def show_graph(graph,high_light_mask:list=[], node_size = 300, width=1,iterations = 50):
    num_nodes = graph.num_nodes()
    scale = num_nodes // 10
    if scale > 3:
        node_size = 300.0 / scale
        width = 10.0 / scale
        if scale > 10:
            iterations = int(scale * 0.5)
    color_list = iter(['r','g','b'])
    nx_graph = dgl.to_networkx(graph,node_attrs=['label'])
    # nx_graph = nx_graph.to_undirected()
    pos = nx.spring_layout(nx_graph,k=0.4,iterations=iterations)
    nx.draw(nx_graph, pos, edge_color='black',node_color = "black",  with_labels=False,
        font_weight='light', node_size= node_size, width= width)
    if len(high_light_mask) > 3:
        print("Too much communities to high light!")
        return
    elif len(high_light_mask) > 0:
        for community_mask in high_light_mask:
            color = next(color_list)
            community_indices = [i for i, value in enumerate(community_mask) if value]
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=community_indices, node_size= node_size, node_color=color)
    file_path = 'CGDM-Im\\history_data\\figure\\graph.png'
    file_path = file_path.replace("\\",os.sep)
    plt.savefig(file_path)
    plt.close()
    # plt.show()

def show_diffusion(diffusion_model:nn.Module,x0,t=1000,device="cuda:0"):
    with torch.no_grad():
        diffusion_model.eval()
        noise = torch.randn_like(x0.to(torch.float32))
        xt = diffusion_model.q_sample(x0, torch.tensor([t],dtype=torch.long).to(device), eps=noise)
        show_tensor_dis(xt)


def show_tensor_dis(input_tensor:torch.Tensor,bins=20, name:str="tensordis"):
    nparray = input_tensor.to("cpu").view(-1).numpy()
    hist1, bin_edges1 = np.histogram(nparray, bins=bins)
    fig1, ax1 = plt.subplots()
    # ax1.hist(bin_edges1[:-1], bin_edges1, weights=hist1, facecolor='skyblue',
    #         alpha=0.7, edgecolor='k')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Counts')

    rects = ax1.bar(bin_edges1[:-1], hist1, width=abs(bin_edges1[1]-bin_edges1[0]), color='skyblue')
    for rect in rects:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., height,
                str(int(height)), ha='center', va='bottom')
    
    plt.tight_layout()
    file_path = "CGDM-Im\\history_data\\figure\\"+name+".jpg"
    file_path = file_path.replace("\\",os.sep)
    plt.savefig(file_path)
    plt.close()

def diffusion_gif(x_frame:torch.Tensor,file_path):
    fig = plt.figure()
    # 创建一个动画对象，将 x_data 和 y_data 作为参数传递给 update 函数
    print("gif generating")
    ani = FuncAnimation(fig, gif_update, frames=len(x_frame), fargs=(x_frame,), interval=200)
    # 保存动画为 GIF 文件
    ani.save(file_path.replace("\\",os.sep), writer='pillow')
    plt.close()
    
def gif_update(frame,x):
    plt.cla() # 清除当前图形
    # 绘制新的图形
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(x[frame][i].squeeze(0)[2:-2,2:-2].to("cpu"), cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(i))
        plt.xticks([])
        plt.yticks([])
 
def evaluation(y_true, y_predict):
    y_true = y_true.to("cpu")
    y_predict = y_predict.to("cpu")
    accuracy=classification_report(y_true, y_predict,output_dict=True)['accuracy']
    s=classification_report(y_true, y_predict,output_dict=True)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    return accuracy,precision,recall,f1_score

def show_detailed_evaluation(y_true, y_predict,dg=4):
    y_true = y_true.to("cpu")
    y_predict = y_predict.to("cpu")
    measure_result = classification_report(y_true, y_predict,digits=dg)
    print('measure_result = \n', measure_result)
    return(measure_result)

def auc_score(logits:torch.Tensor,targets:torch.Tensor,num_classes):
    assert logits.shape[1] == num_classes, "logits must shape as [samples, num_classes]"
    logits = logits.softmax(dim=1).to("cpu").detach().numpy()
    targets = targets.to("cpu").detach().numpy()
    score = roc_auc_score(targets,logits,average='macro',multi_class="ovr")
    return score

def extract_config(model:nn.Module):
    config = {}
    for k, v in model.__dict__.items():
        # 跳过内部属性、方法、tensor、layer 等
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        # 只保存基础类型
        if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            config[k] = v
    return config

def save(model,file_path:str):
    try:
        file_path = file_path.replace("\\",os.sep)
        config = extract_config(model)  # 自动提取结构信息
        ckpt = {
            "config": config,
            "state_dict": model.state_dict()
        }
        torch.save(ckpt, file_path)
    except FileNotFoundError as fnf:
        print("MODEL IS NOT SAVED!")

def save_classification_results(labels: torch.Tensor, predictions: torch.Tensor, output_file: str):
    """
    保存模型的分类结果到CSV文件。

    参数:
        labels (torch.Tensor): 真实标签张量。
        predictions (torch.Tensor): 模型预测结果张量。
        output_file (str): 输出文件名。
    """
    # 确保输入是1D张量
    labels = labels.cpu().flatten().tolist()
    predictions = predictions.cpu().flatten().tolist()

    # 写入CSV文件
    with open(output_file, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Prediction'])  # 写入表头
        writer.writerows(zip(labels, predictions))  # 写入标签和预测

    print(f"结果已保存至 {output_file}")

def plot_performance_with_error(csv_file: str, output_image: str):
    """
    从CSV文件读取数据并生成带误差范围的折线图。

    参数:
        csv_file (str): 输入的CSV文件路径。
        output_image (str): 输出图像保存路径。
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    # 提取参数imb_ratio
    imb_ratios = data.iloc[:, 0]
    
    plt.rcParams["font.family"] = '/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf'  # 设置字体
    plt.rcParams['font.size'] = 16                # 设置字号为 24
    # 初始化图像
    plt.figure(figsize=(10, 6))
    colors = cm.get_cmap("tab10")  # 获取 tab10 调色板

    # 解析每列模型数据

    for idx, col in enumerate(data.columns[1:]):
        # method_name = data[col][0]  # 获取方法名
        # 提取性能和误差
        values = data[col][:]
        f1_scores = []
        error_ranges = []

        for val in values:
            match = re.match(r"([\d\.]+)\+\-([\d\.]+)", str(val))
            if match:
                f1 = float(match.group(1))
                error = float(match.group(2))
                f1_scores.append(f1)
                error_ranges.append(error)
        
        # 转换为NumPy数组
        f1_scores = np.array(f1_scores)
        error_ranges = np.array(error_ranges)
        # 绘制带误差范围的折线图
        color = colors(idx % 10)  # 从 tab10 获取颜色
        plt.plot(imb_ratios, f1_scores, label=col, marker='o', color=color)
        plt.fill_between(imb_ratios, f1_scores - error_ranges, f1_scores + error_ranges, alpha=0.2, color=color)

    # 设置图例和轴标签
    plt.xlabel("imbalance ratio")
    plt.ylabel("F1 Score")
    # plt.title("Model Performance with Error Ranges")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(ticks=imb_ratios, labels=imb_ratios.astype(int))  # 设置 x 轴刻度为整数值
    # 保存图像
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"图像已保存至 {output_image}")

def plot_metrics(x, y1, y2, xlabel, ylabel, title, label1, label2, marker1='o', marker2='s', markersize=3):
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线，markersize 用于控制点的大小
    plt.plot(x, y1, label=label1, marker=marker1, markersize=markersize, linestyle='-', zorder=1)
    plt.plot(x, y2, label=label2, marker=marker2, markersize=markersize, linestyle='-', zorder=1)
    
    # 标出最高值
    max_y1 = max(y1)
    max_y2 = max(y2)
    max_y1_idx = y1.index(max_y1)
    max_y2_idx = y2.index(max_y2)
    
    # 使用 scatter 绘制最高点，设置 zorder 确保最高点不被遮挡，edgecolor 设置黑色边框
    plt.scatter(max_y1_idx, max_y1, color='blue', label=f'Max {label1}', s=100, edgecolor='black', zorder=3)
    plt.scatter(max_y2_idx, max_y2, color='red', label=f'Max {label2}', s=100, edgecolor='black', zorder=3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(".\\history_data\\figure\\".replace("\\",os.sep)+title.replace(" ","_")+".png")
    plt.close()

# 绘制 val_acc_f1 和 test_acc_f1 的变化曲线
def plot_val_test_acc_f1(val_acc_f1_list, test_acc_f1_list,title='val_acc_f1 vs test_acc_f1 over Epochs'):
    epochs = list(range(1, len(val_acc_f1_list) + 1))
    plot_metrics(epochs, val_acc_f1_list, test_acc_f1_list, 
                 xlabel='Epoch', ylabel='Value', 
                 title=title, 
                 label1='val_acc_f1', label2='test_acc_f1')

# 绘制 val_f1 和 val_acc 的变化曲线
def plot_val_acc_f1(val_f1_list, val_acc_list, title='val_f1 vs val_acc over Epochs'):
    epochs = list(range(1, len(val_f1_list) + 1))
    plot_metrics(epochs, val_f1_list, val_acc_list, 
                 xlabel='Epoch', ylabel='Value', 
                 title=title, 
                 label1='val_f1', label2='val_acc')

# 绘制 tmp_test_acc 和 tmp_test_f1 的变化曲线
def plot_tmp_test_acc_f1(tmp_test_acc_list, tmp_test_f1_list, title='tmp_test_acc vs tmp_test_f1 over Epochs'):
    epochs = list(range(1, len(tmp_test_acc_list) + 1))
    plot_metrics(epochs, tmp_test_acc_list, tmp_test_f1_list, 
                 xlabel='Epoch', ylabel='Value', 
                 title=title, 
                 label1='tmp_test_acc', label2='tmp_test_f1')