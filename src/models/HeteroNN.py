import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv

class GAT(torch.nn.Module):

    def __init__(self, features, hidden, heads):

        super(GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads) 
        self.gat2 = GATConv(hidden*heads, hidden)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x , F.log_softmax(x)

class HeteroSAGE(torch.nn.Module):

    def __init__(self, metadata, hidden_channels, num_layers):

        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=0.15)
        self.bn = torch.nn.BatchNorm1d(hidden_channels, momentum=0.01)
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1,-1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        layer = 1
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            if(layer < 2):
                for item in list(x_dict.keys()):
                    x_dict[item] = self.dropout(x_dict[item])
                    x_dict[item] = self.bn(x_dict[item])
                layer = layer + 1
        return x_dict


class HeteroGAT(torch.nn.Module):

    def __init__(self, metadata, hidden_channels, num_layers, num_heads=4):

        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=0.15)
        self.bn = torch.nn.BatchNorm1d(num_heads * hidden_channels, momentum=0.01)
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        layer = 1
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            if(layer < 2):
                for item in list(x_dict.keys()):
                    x_dict[item] = self.dropout(x_dict[item])
                    x_dict[item] = self.bn(x_dict[item])
                layer += 1  
        return x_dict