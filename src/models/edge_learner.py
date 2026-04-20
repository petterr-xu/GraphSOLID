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

    def _project(self, node_type, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.node_projectors[node_type](z)

    def score_aligned_pairs(self, src_emb, dst_emb, edge_type):
        src_type, _, dst_type = edge_type
        rel_key = "__".join(edge_type)
        z_src = self._project(src_type, src_emb)
        z_dst = self._project(dst_type, dst_emb)
        z_src_rel = self.rel_weights[rel_key](z_src)
        return (z_src_rel * z_dst).sum(dim=-1)

    def score_query_candidates(self, query_emb, candidate_emb, edge_type, query_is_src=True):
        src_type, _, dst_type = edge_type
        rel_key = "__".join(edge_type)
        if query_is_src:
            query_proj = self.rel_weights[rel_key](self._project(src_type, query_emb))
            candidate_proj = self._project(dst_type, candidate_emb)
        else:
            query_proj = self._project(dst_type, query_emb)
            candidate_proj = self.rel_weights[rel_key](self._project(src_type, candidate_emb))
        scores = query_proj @ candidate_proj.t()
        if scores.size(0) == 1:
            return scores.squeeze(0)
        return scores

    def forward(self, z_dict, edge_index, edge_type):
        """
        z_dict: 节点嵌入字典 {type: tensor}
        edge_index: 当前预测的边索引
        edge_type: 当前预测的边类型三元组 ('src', 'rel', 'dst')
        """
        src_type, _, dst_type = edge_type
        src_emb = z_dict[src_type][edge_index[0]]
        dst_emb = z_dict[dst_type][edge_index[1]]
        return self.score_aligned_pairs(src_emb, dst_emb, edge_type)

class BudgetPredictor(nn.Module):
    def __init__(self, num_feat, num_cls, num_edge_types, layers=2, drop=0.3):
        super().__init__()
        self.num_cls = num_cls
        self.mlp = mlp.MLP(num_feat + num_cls, num_edge_types, layers=layers, drop=drop)
        self.act = nn.ReLU()

    def forward(self, node_emb, soft_labels, inference=False):
        if node_emb.dim() == 1:
            node_emb = node_emb.unsqueeze(0)
        if soft_labels.dim() == 1:
            soft_labels = soft_labels.unsqueeze(0)

        if node_emb.size(0) != soft_labels.size(0):
            raise ValueError(
                f"node_emb batch size {node_emb.size(0)} != soft_labels batch size {soft_labels.size(0)}"
            )
        if soft_labels.size(-1) != self.num_cls:
            raise ValueError(
                f"soft_labels last dim must be {self.num_cls}, got {soft_labels.size(-1)}"
            )

        budget_scores = self.mlp(torch.cat([node_emb, soft_labels], dim=-1))
        budgets = self.act(budget_scores)
        if inference:
            budgets = torch.round(budgets)
        return budgets


class AdaptiveEdgePredictor(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        node_dim_dict,
        n_hid,
        num_cls,
        budget_layers=2,
        drop=0.3,
    ):
        super().__init__()
        self.edge_types = list(edge_types)
        self.edge_type_to_idx = {"__".join(edge_type): i for i, edge_type in enumerate(self.edge_types)}
        self.edge_predictor = HeteroEdgePredicter(node_types, edge_types, node_dim_dict, n_hid)
        self.budget_predictor = BudgetPredictor(
            num_feat=n_hid,
            num_cls=num_cls,
            num_edge_types=len(self.edge_types),
            layers=budget_layers,
            drop=drop,
        )

    def forward(
        self,
        gen_emb,
        soft_labels,
        candidate_emb,
        edge_type,
        generated_node_type=None,
        inference=False,
    ):
        rel_key = "__".join(edge_type)
        if rel_key not in self.edge_type_to_idx:
            raise KeyError(f"Unknown edge type: {edge_type}")

        src_type, _, dst_type = edge_type
        if generated_node_type is None:
            generated_node_type = src_type
        if generated_node_type not in (src_type, dst_type):
            raise ValueError(
                f"generated_node_type must be one of ({src_type}, {dst_type}), got {generated_node_type}"
            )

        query_is_src = generated_node_type == src_type
        budgets = self.budget_predictor(gen_emb, soft_labels, inference=inference)
        edge_budget = budgets[..., self.edge_type_to_idx[rel_key]]
        scores = self.edge_predictor.score_query_candidates(
            query_emb=gen_emb,
            candidate_emb=candidate_emb,
            edge_type=edge_type,
            query_is_src=query_is_src,
        )
        return edge_budget, scores
