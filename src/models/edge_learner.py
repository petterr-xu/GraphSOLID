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


# def edge_predictor(z, edge_index):
#     edge_embeddings = z[edge_index[0]] * z[edge_index[1]]
#     return torch.sigmoid(edge_embeddings.sum(dim=1))

# def get_link_labels(pos_edge_index, neg_edge_index):
#     num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
#     link_labels = torch.zeros(num_links, dtype=torch.float)
#     link_labels[:pos_edge_index.size(1)] = 1.
#     return link_labels

# def train(data, model, optimizer):
#     model.train()

#     neg_edge_index = negative_sampling(
#         edge_index=data.train_pos_edge_index,
#         num_nodes=data.num_nodes,
#         num_neg_samples=data.train_pos_edge_index.size(1))

#     optimizer.zero_grad()
#     z = model.encode(data.x, data.train_pos_edge_index)
#     link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
#     link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
#     loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
#     loss.backward()
#     optimizer.step()

#     return loss