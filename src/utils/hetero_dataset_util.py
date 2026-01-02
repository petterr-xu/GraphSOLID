import json
import scipy.io as sio
import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

class HeteroGraphContext:
    """
    自定义的图数据上下文对象，封装了元数据和 PyG 异构图。
    """
    def __init__(self, metadata, pyg_graph):
        self.meta = metadata      # 原始元数据字典
        self.g = pyg_graph        # PyG HeteroData 对象
        self.name = metadata.get('dataset_name', 'Unknown')
        self.target_node = metadata.get('target_node')

    def __repr__(self):
        return f"<FraudGraphContext: {self.name} | Nodes: {self.g.node_types} | Edges: {self.g.edge_types}>"

    @property
    def n_classes(self):
        # 自动获取目标节点的类别数
        return int(self.g[self.target_node].y.max()) + 1

    @property
    def n_features(self):
        # 自动获取目标节点的特征维度
        return self.g[self.target_node].x.shape[1]

class GraphDataLoader:
    """
    负责反序列化：读取配置 -> 加载 MAT -> 封装 Context
    """
    @staticmethod
    def load_from_config(config_path, mat_path, device=None):
        # 1. 读取元数据文件
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 2. 加载原始 MAT 数据
        mat = sio.loadmat(mat_path)
        data = HeteroData()
        
        # 3. 动态转化节点
        for node_type, config in meta['nodes'].items():
            feat = mat[config['feat_key']]
            # 自动处理稀疏/稠密矩阵
            x = torch.from_numpy(feat.todense() if hasattr(feat, 'todense') else feat).float()
            data[node_type].x = x
            
            if config.get('label_key') in mat:
                y = torch.from_numpy(mat[config['label_key']].flatten()).long()
                data[node_type].y = y
        
        # 4. 动态转化边
        for mat_key, edge_triplet in meta['edges'].items():
            if mat_key in mat:
                # 确保 edge_triplet 是 tuple (src, rel, dst)
                triplet = tuple(edge_triplet)
                row, col = mat[mat_key].nonzero()
                data[triplet].edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        # 5. 自动执行基础转换 (如划分 Mask)
        # 这里默认进行 40/20/40 划分，后续也可以通过 context.g 修改
        transform = T.RandomNodeSplit(num_val=0.2, num_test=0.4)
        data = transform(data)
        # 手动划分验证边 
        for etype in data.edge_types:
            data = manual_split_hetero_edges(data, etype)
        if device:
            data = data.to(device)
        # 6. 封装并返回
        return HeteroGraphContext(meta, data)

def manual_split_hetero_edges(data, edge_type, val_ratio=0.1, test_ratio=0.1, max_train_edges=100000):
    """
    将异构边划分为 训练/验证/测试 三个集合
    """
    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)
    
    # 1. 计算各集合数量
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train_all = num_edges - num_val - num_test
    
    # 2. 随机打乱并切分
    perm = torch.randperm(num_edges)
    
    val_indices = perm[:num_val]
    test_indices = perm[num_val : num_val + num_test]
    train_indices_all = perm[num_val + num_test:]
    
    # 3. 限制训练边数以提速
    if max_train_edges is not None and train_indices_all.size(0) > max_train_edges:
        train_indices = train_indices_all[:max_train_edges]
    else:
        train_indices = train_indices_all
        
    # 4. 赋值给 Data 对象
    data[edge_type].train_pos_edge_index = edge_index[:, train_indices]
    data[edge_type].val_pos_edge_index = edge_index[:, val_indices]
    data[edge_type].test_pos_edge_index = edge_index[:, test_indices]
    
    return data