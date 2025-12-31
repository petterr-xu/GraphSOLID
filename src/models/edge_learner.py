import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mlp,sage

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        score = F.sigmoid(self.W2(F.relu(self.W1(h))))
        return {"score": score.squeeze(1)}
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]

class EdgePredicter(nn.Module):
    def __init__(self,n_emb,drop=0.3) -> None:
        super().__init__()
        self.linear = nn.Linear(n_emb,n_emb)
        # self.linear = nn.Sequential()
        # self.n_hid = n_emb // 2
        # self.linear.add_module("linear1-{}".format(self.n_hid),nn.Linear(n_emb,self.n_hid))
        # self.linear.add_module("act1",nn.ReLU())
        # self.linear.add_module("drop1",nn.Dropout(drop))
        # self.linear.add_module("linear2-{}".format(self.n_hid),nn.Linear(self.n_hid,self.n_hid))
    def forward(self,z,edge_index):
        emb = self.linear(z)
        scores = (emb[edge_index[0]] * emb[edge_index[1]]).sum(dim=-1)
        return scores
    def decode(self,z):
        emb = self.linear(z)
        prob_adj = emb @ emb.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class HeteroEdgePredicter(nn.Module):
    def __init__(self, node_types, edge_types, node_dim_dict, n_hid):
        """
        node_types: 节点类型列表, ['review', 'user', ...]
        edge_types: 边三元组列表, [('review', 'rur', 'review'), ...]
        node_dim_dict: 每个节点类型的输入维度
        n_hid: 映射后的统一隐空间维度
        """
        super().__init__()

        # 维度对齐层：为每种节点类型分配一个 Linear
        self.node_projectors = nn.ModuleDict({
            node_type: nn.Linear(node_dim_dict[node_type], n_hid)
            for node_type in node_types
        })
        
        # 关系特异层：为每种关系分配一个权重矩阵
        # 即使 src 和 dst 一样，通过不同的 relation 权重也能区分不同边
        self.rel_weights = nn.ModuleDict({
            "__".join(edge_type): nn.Linear(n_hid, n_hid, bias=False)
            for edge_type in edge_types
        })

    def forward(self, z_dict, edge_index, edge_type):
        """
        z_dict: 节点嵌入字典 {type: tensor}
        edge_index: 当前预测的边索引
        edge_type: 当前预测的边类型三元组 ('src', 'rel', 'dst')
        """
        src_type, rel_name, dst_type = edge_type
        rel_key = "__".join(edge_type)
        
        # 1. 取出对应的嵌入并投影到统一维度
        z_src = self.node_projectors[src_type](z_dict[src_type])
        z_dst = self.node_projectors[dst_type](z_dict[dst_type])
        
        # 2. 应用关系变换矩阵
        # 这里模拟了关系对嵌入的影响，使得不同关系下的同一对节点得分不同
        z_src_rel = self.rel_weights[rel_key](z_src)
        
        # 3. 计算点积得分
        scores = (z_src_rel[edge_index[0]] * z_dst[edge_index[1]]).sum(dim=-1)
        return scores