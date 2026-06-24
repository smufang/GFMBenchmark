from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F
from torch_geometric.nn.conv import FAConv
from torch_geometric.nn import MessagePassing, SAGEConv, GATConv, GCNConv, GINConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


class FALayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1, epsilon=0.1,
                 activation="relu", normalize=True, add_self_loops=True, cached=False):
        super(FALayers, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "elu":
            self.act_fn = F.elu
        elif activation == "none":
            self.act_fn = nn.Identity()
        else:
            self.act_fn = F.relu  # default

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(FAConv(hidden_dim, eps=epsilon, dropout=0.0,
                                     cached=cached, add_self_loops=add_self_loops,
                                     normalize=normalize))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            self.input_proj.bias.data.fill_(0.0)
    
    def reset_parameters(self):
        self.init_weights()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, LP=False, prompt_layers=None, name=None):
        if prompt_layers:
            assert len(prompt_layers) == self.num_layers
        h = self.input_proj(x)
        h = self.act_fn(h)
        raw = h
        for i in range(self.num_layers):
            h = self.convs[i](h, raw, edge_index)
            if prompt_layers:
                h = prompt_layers[i](h, name)
            if LP:
                h = self.bns[i](h)
                h = self.dropout(h)
        return h


class MySAGEConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Union[Tensor, None] = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor, xe: Optional[Tensor])
        out = self.propagate(edge_index, x=x, size=size, xe=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, xe) -> Tensor:
        return (x_j + xe).relu()

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation, num_layers, backbone='fagcn',
                 normalize='none', dropout=0.0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize

        self.activation = activation()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * num_layers

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == 'fagcn':
                self.layers.append(FALayers(in_dim, out_dim))
            if backbone == 'sage':
                self.layers.append(MySAGEConv(in_dim, out_dim, aggr='mean', normalize=False, root_weight=True))
            elif backbone == 'gat':
                self.layers.append(GATConv(in_dim, out_dim, heads=1))
            elif backbone == 'gcn':
                self.layers.append(GCNConv(in_dim, out_dim, ))
            elif backbone == 'gin':
                self.layers.append(GINConv(nn.Linear(in_dim, out_dim)))
            self.norms.append(nn.BatchNorm1d(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return z

    def encode(self, x, edge_index, edge_attr=None):
        z = x

        for i, conv in enumerate(self.layers):
            if self.backbone == 'fagcn':
                z = conv(z, edge_index)
            else:
                z = conv(z, edge_index, edge_attr)
            if self.normalize != 'none':
                z = self.norms[i](z)
            if i < self.num_layers - 1:
                z = self.activation(z)
                z = self.dropout(z)
        return z


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def __init__(self, hidden_dim=None, output_dim=None):
        super().__init__()
        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
