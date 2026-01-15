import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from . import gcn, sage, gat, mlp

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z_list):
        # z_list: [N_views, N_nodes, D]
        z = torch.stack(z_list, dim=1) # [N, N_views, D]
        
        # 计算视图重要性得分
        # 对每个节点在每个视图下的表示进行投影，然后对所有节点取平均
        s = self.project(z).mean(0)    # [N_views, 1]
        beta = torch.softmax(s, dim=0) # [N_views, 1]
        
        # 扩展维度进行加权求和
        beta = beta.expand((z.shape[0], -1, -1)) # [N, N_views, 1]
        z_final = (beta * z).sum(1)              # [N, D]
        
        return z_final, beta[0].flatten()

class MultiViewGNN(nn.Module):
    def __init__(self, net, n_features, n_classes, hidden_dim, view_names):
        super(MultiViewGNN, self).__init__()
        self.view_names = view_names
        if net is 'GCN':
            GNNLayer = gcn
        elif net is 'SAGE':
            GNNLayer = sage
        elif net is 'GAT':
            GNNLayer = gat
        else:
            raise NotImplementedError("Not Implemented Architecture: "+net)
        
        self.encoders = nn.ModuleDict({
            name: GNNLayer(n_features, hidden_dim) for name in view_names
        })
        
        # 语义融合层
        self.semantic_fusion = SemanticAttention(hidden_dim)
        
        # 分类头
        self.classifier = mlp.MLP(hidden_dim, n_classes, 2)

    def forward(self, views_dict):
        """
        views_dict: 由 ctx.get_views() 返回的字典 {view_name: Data_object}
        """
        z_list = []
        
        for name in self.view_names:
            data = views_dict[name]
            # edge_index = self.denoise(data.edge_index) 
            z = self.encoders[name](data.x, data.edge_index)
            z = F.relu(z)
            z_list.append(z)
            
        # 基于注意力的嵌入融合
        combined_z, att_weights = self.semantic_fusion(z_list)
        
        out = self.classifier(combined_z)
        
        return out, att_weights