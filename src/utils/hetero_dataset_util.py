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
        if device:
            data = data.to(device)
        # 6. 封装并返回
        return HeteroGraphContext(meta, data)