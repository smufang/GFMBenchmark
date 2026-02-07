import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from layers import *


class Pretrain(nn.Module):
    """
    This model does not support pretraining.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This model does not have a pretraining stage. "
            "The Pretrain module is a placeholder and must not be instantiated."
        )

class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super(Downstream, self).__init__()
        input_dim = configs.input_dim
        if input_dim == -1:
            input_dim = dataset.x.size(1)
        self.num_ntypes = int(dataset.node_type.max().item()) + 1
        self.num_etypes = int(dataset.edge_type.max().item()) + 1
        self.in_dim = [input_dim] * self.num_ntypes  # Should be list per node type

        hidden_dim = configs.hidden_dim
        num_layers = configs.num_layers
        num_heads = configs.num_heads 
        # Ideally num_heads is a list [8, 8, ..., 1], if int provided, we convert to list
        if isinstance(num_heads, int):
            num_heads = [num_heads] * (num_layers + 1)

        dropout = configs.dropout
        edge_dim = configs.edge_dim
        activation = configs.activation
        alpha = configs.alpha
        slope = configs.beta # negative slope for leaky relu
        
        self.task_name = configs.task_name
        self.num_classes = configs.num_classes

        # --- Encoder Initialization ---
        self.gnn_enc = SimpleHGNLayers(
            in_dims=self.in_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            num_etypes=self.num_etypes,
            edge_dim=edge_dim,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=activation,
            negative_slope=slope,
            alpha=alpha
        )

    def embedding_forward(self, data):
        """
        Expects data to be a Homogeneous Graph object (e.g. from HeteroData.to_homogeneous())
        containing: data.x, data.edge_index, data.edge_type, data.node_type
        """
        embed = self.gnn_enc(data.x, data.edge_index, data.edge_type, data.node_type)
        return embed

    def compute_predictions(self, embed, labels=None, is_train=True):
        """
        embed: [num_items, num_classes]
        labels: [num_items] only used if is_train=True
        """
        return F.log_softmax(embed, dim=1)

    def node_classification(self, data, labels, node_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        select_node_embed = node_embed[node_idx]  # [num_nodes, num_classes] -> selected
        return self.compute_predictions(select_node_embed, labels, is_train)

    def edge_classification(self, data, edge_type, edge_idx, is_train=True): pass

    def graph_classification(self, data, graph_labels, graph_idx, is_train=True): pass

    def forward(self, data, labels, idx, is_train=True):
        if self.task_name == 'node':
            return self.node_classification(data, labels, idx, is_train)
        elif self.task_name == 'edge':
            raise NotImplementedError("Edge classification is not implemented yet.")
        elif self.task_name == 'graph':
            raise NotImplementedError("Graph classification is not implemented yet.")
        else:
            raise ValueError("Invalid task name. Use 'node', 'edge', or 'graph'.")