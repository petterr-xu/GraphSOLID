import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import VNG_utils
from utils.config import GraphEncoderConfig
from . import gnn

class Decoder(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)
    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))
        return adj_out
    
def smote(x, k=4, beta=1.0):
    n, _ = x.shape
    num_samples_to_generate = int(beta * n)
    dist_matrix = torch.cdist(x, x, p=2)
    knn_indices = dist_matrix.topk(k=k+1, largest=False).indices[:, 1:]
    synthetic_samples = []
    for _ in range(num_samples_to_generate):
        i = torch.randint(0, n, (1,)).item()
        nn_index = torch.randint(0, k, (1,)).item()
        neighbor = x[knn_indices[i, nn_index]]
        lam = torch.rand(1)
        synthetic_sample = x[i] + lam * (neighbor - x[i])
        synthetic_samples.append(synthetic_sample)
    synthetic_samples = torch.stack(synthetic_samples)
    augmented_x = torch.cat([x, synthetic_samples], dim=0)
    
    return augmented_x



def train_graph_smote(gnn_type):
    graph = VNG_utils.load_cora_raw().to("cuda:0")
    if gnn_type == "sage":
        graph_encoder = gnn.GraphSAGE(1433,512,layers=1,drop=0.1)
    elif gnn_type == "gcn":
        graph_encoder = gnn.GCN(1433,512,layers=1,drop=0.1)
    
    pass