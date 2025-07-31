import torch.nn as nn
import torch.nn.functional as F

from . import mlp,gcn,sage,gat

class GNN_classifier(nn.Module):
    def __init__(self, net, nfeat, nhid, nclass, nlayer=1, dropout=0.5):
        super(GNN_classifier, self).__init__()
        if net == 'GCN':
            self.gnn = gcn.GCN_single(nfeat, nhid, nhid, 1, dropout)
        elif net == "SAGE":
            self.gnn = sage.GraphSAGE_single(nfeat, nhid, nhid, 1, dropout)
        elif net == "GAT":
            self.gnn = gat.GAT_single(nfeat, nhid, nhid, 1, dropout)
        else:
            raise NotImplementedError("Not Implemented Architecture!")
        self.classifier = mlp.MLP(nhid,nclass,nlayer,dropout)
        self.reg_params = list(self.gnn.parameters()) + list(self.classifier.layers.parameters())
        self.non_reg_params = self.classifier.fc.parameters()

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.gnn(x, edge_index, edge_weight)
        out = self.classifier(x)
        return out