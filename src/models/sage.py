"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/sage_conv.py
"""
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add

import numpy as np

class MeanAggregation(MessagePassing):
    def __init__(self):
        super(MeanAggregation, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.temp_weight = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.temp_weight.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.temp_weight(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(nn.Module):
    def __init__(self, in_feat, nhid, out_feat,nlayer=3, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feat, nhid)
        self.conv2 = SAGEConv(nhid, out_feat)
        self.convx = nn.ModuleList([SAGEConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index,edge_weight))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)

        return x

class GraphSAGE_res(nn.Module):
    def __init__(self, in_feat, nhid, out_feat,nlayer=3, dropout=0.3):
        super(GraphSAGE_res, self).__init__()
        self.conv1 = SAGEConv(in_feat, nhid)
        self.conv2 = SAGEConv(nhid, out_feat)
        self.convx = nn.ModuleList([SAGEConv(nhid, nhid) for _ in range(nlayer-2)])
        self.short_cut = nn.Linear(in_feat,out_feat)
        self.dropout_p = dropout
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = list(self.short_cut.parameters()) + list(self.conv2.parameters())

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        raw_x = x
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index,edge_weight))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        x = x + self.short_cut(raw_x)
        return x

class GraphSAGE_single(nn.Module):
    def __init__(self, in_feat, nhid, out_feat, nlayer=1, dropout=0.5):
        super(GraphSAGE_single, self).__init__()
        self.conv1 = SAGEConv(in_feat, out_feat)
        # self.short_cut = nn.Linear(in_feat,out_feat)
        self.dropout_p = dropout
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        # raw_x = x
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = x + self.short_cut(raw_x)
        return x