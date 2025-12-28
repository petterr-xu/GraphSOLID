import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv, SAGEConv, HeteroConv

from . import mlp

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

class RGCN(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, num_heads=4):
        super(RGCN, self).__init__()
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=0.15)
        self.bn = torch.nn.BatchNorm1d(num_heads * hidden_channels, momentum=0.01)
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GraphConv((-1,-1), hidden_channels, add_self_loops=False)
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

class HeteroGNN_classifier(nn.Module):
    def __init__(self, metadata, nhid, nclass, nlayer=1, dropout=0.5, target_node="review"):
        """
        metadata: 异构图元数据 (ctx.g.metadata())
        nhid: 隐藏层维度
        nclass: 类别总数
        nlayer: MLP 分类器的层数
        target_node: 需要进行分类的目标节点类型
        """
        super(HeteroGNN_classifier, self).__init__()
        
        # 定义骨干网络：异构 SAGE
        # 这里的 num_layers 指的是 GNN 的层数
        self.gnn = HeteroSAGE(metadata, nhid, num_layers=2)
        
        # 定义分类头：MLP
        self.classifier = mlp.MLP(nhid, nclass, nlayer, dropout)        
        self.target_node = target_node
        
        self.reg_params = list(self.gnn.parameters()) + list(self.classifier.parameters())

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: 节点特征字典
        edge_index_dict: 边索引字典
        """
        # 1. 通过 HeteroSAGE 获取所有节点的 Embedding 字典
        # out_dict: {node_type: [num_nodes, nhid]}
        out_dict = self.gnn(x_dict, edge_index_dict)
        
        # 2. 提取目标节点的嵌入
        target_emb = out_dict[self.target_node]
        
        # 3. 通过 MLP 分类器得到最终预测
        logits = self.classifier(target_emb)
        
        return logits