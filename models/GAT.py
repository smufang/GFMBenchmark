import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from layers import *
from utils.similarity import cosine_similarity
from torch_geometric.nn.models import GAT


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
        hidden_dim = configs.hidden_dim
        num_layers = configs.num_layers
        num_heads = configs.num_heads

        self.task_name = configs.task_name
        self.num_classes = configs.num_classes
        self.temperature = configs.temperature
        class_dim = hidden_dim*2 if self.task_name == 'edge' else hidden_dim
        self.class_prototypes = torch.FloatTensor(self.num_classes, class_dim)

        self.gnn_enc = GAT(in_channels=input_dim, 
                           hidden_channels=hidden_dim, 
                           num_layers=num_layers, 
                           heads=num_heads,
                           act='relu')

        self.using_projection = configs.using_projection
        self.projection_head = nn.Linear(class_dim, self.num_classes)

    def embedding_forward(self, data):
        embed = self.gnn_enc(data.x, data.edge_index)
        return embed

    def compute_predictions(self, embed, labels=None, is_train=True):
        """
        embed: [num_items, hidden_dim]
        labels: [num_items] only used if is_train=True
        """
        if self.using_projection:
            similarity = self.projection_head(embed) # [num_items, num_classes]
        else:
            if is_train and labels is not None:
                self.class_prototypes = torch_scatter.scatter(src=embed, index=labels, dim=0,
                            reduce='mean', dim_size=self.num_classes)  # [num_classes, hidden_dim]

            similarity = cosine_similarity(embed, self.class_prototypes, temperature=self.temperature, eps=1e-10)  # [num_items, num_classes]
            
        if is_train:
            return similarity
        else:
            return F.softmax(similarity, dim=1)

    def node_classification(self, data, labels, node_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        select_node_embed = node_embed[node_idx]  # [num_nodes, hidden_dim] -> selected
        return self.compute_predictions(select_node_embed, labels, is_train)

    def edge_classification(self, data, edge_type, edge_idx, is_train=True):
        # edge_idx: [2, num_task_edges]
        node_embed = self.embedding_forward(data)
        select_src_idx, select_dst_idx = edge_idx
        select_edge_embed = torch.cat([node_embed[select_src_idx], node_embed[select_dst_idx]], dim=-1)
        return self.compute_predictions(select_edge_embed, edge_type, is_train)

    def graph_classification(self, data, graph_labels, graph_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        graph_embed = torch_scatter.scatter(node_embed, data.batch, dim=0, reduce='mean')
        selected_graph_embed = graph_embed[graph_idx]
        return self.compute_predictions(selected_graph_embed, graph_labels, is_train)

    def forward(self, data, labels, idx, is_train=True):
        if self.task_name == 'node':
            return self.node_classification(data, labels, idx, is_train)
        elif self.task_name == 'edge':
            return self.edge_classification(data, labels, idx, is_train)
        elif self.task_name == 'graph':
            return self.graph_classification(data, labels, idx, is_train)
        else:
            raise ValueError("Invalid task name. Use 'node', 'edge', or 'graph'.")