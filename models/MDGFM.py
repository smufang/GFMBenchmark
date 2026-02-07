import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_geometric
from torch_geometric.utils import to_undirected
from layers import *
from utils.knn import knn_fast
from utils.similarity import cosine_similarity


class Pretrain(nn.Module):
    def __init__(self, configs):
        super(Pretrain, self).__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim
        num_layers = configs.num_layers
        num_heads = configs.num_heads
        backbone = configs.backbone
        combinetype = configs.combinetype
        activation = configs.activation
        dropout = configs.dropout
        self.domain_id = configs.pretrain_domain_id
        self.temperature = configs.temperature
        self.drop_percent = configs.drop_percent
        self.k = configs.k

        if backbone == "gcn":
            self.gnn_enc = GCNLayers(
                input_dim,
                hidden_dim,
                num_layers,
                dropout=dropout,
                activation=activation,
                normalize=False,
                add_self_loops=False,
            )
        elif backbone == "gat":
            self.gnn_enc = GATLayers(
                input_dim,
                hidden_dim,
                num_layers,
                num_heads=num_heads,
                concat=True,
                bias=True,
                dropout=dropout,
                activation=activation,
            )

        self.feature_prompt_layer = AlignPrompt(
            input_dim, domain_id=self.domain_id, combinetype=combinetype
        )
        self.shared_token = TextPrompt(input_dim, combinetype=combinetype)
        # self.balance_token = TextPrompt(input_dim * 2, combinetype=combinetype)
        self.balance_token = AlignPrompt(
            input_dim * 2, domain_id=self.domain_id, combinetype=combinetype
        )  # different for different domains
        self.sigmod = nn.ELU()

    def get_domain_ids(self):
        return self.domain_id

    @staticmethod
    def calc_lower_bound(z_1, z_2, pos_edges, temperature=1.0, eps=1e-8):
        """
        Compute symmetric multi-positive weighted InfoNCE loss using sparse positive edges.
        Args:
            z_1: Tensor of shape [num_nodes, hidden_dim], embeddings from view 1
            z_2: Tensor of shape [num_nodes, hidden_dim], embeddings from view 2
            pos_edges: list of tuples (edge_index, edge_weight)
                - edge_index: [2, num_edges] tensor with row, col of positive edges
                - edge_weight: [num_edges] tensor of positive edge weights (optional, can be None)
        """
        device = z_1.device
        z_2 = z_2.to(device)
        similarity = cosine_similarity(
            z_1, z_2, temperature=temperature, eps=eps
        )  # [num_nodes, num_nodes]
        prob = F.softmax(similarity, dim=-1)

        def compute_weighted_loss(prob_matrix, edge_index, edge_weight=None):
            src, dst = edge_index.to(device)
            pos_prob = prob_matrix[src, dst]

            if edge_weight is None:
                edge_weight = torch.ones_like(pos_prob, device=device)
            else:
                edge_weight = edge_weight.to(device)

            # accumulate weighted probabilities per node
            loss_per_node = torch.zeros(prob_matrix.size(0), device=device)
            loss_per_node.scatter_add_(0, src, pos_prob * edge_weight)
            loss_per_node = torch.clamp(loss_per_node, min=eps)
            return -torch.log(loss_per_node).mean()

        # compute symmetric loss
        total_loss = torch.tensor(0.0, device=device)
        for edge_index, edge_weight in pos_edges:
            loss_fwd = compute_weighted_loss(prob, edge_index, edge_weight)
            loss_bwd = compute_weighted_loss(prob.T, edge_index, edge_weight)
            total_loss += (loss_fwd + loss_bwd) / 2

        return total_loss

    def multi_reproduce_graph(self, x, k, batch, block_size=1024, eps=1e-8):
        """
        Constructs edge_index and edge_weight for each batch_graph separately.
        """
        all_edge_indices = []
        all_edge_weights = []

        for batch_id in batch.unique():
            batch_mask = batch == batch_id
            batch_x = x[batch_mask]
            batch_node_idx = torch.nonzero(batch_mask, as_tuple=True)[0]

            edge_index, edge_weight = knn_fast(batch_x, k, block_size, eps=eps)
            edge_index = batch_node_idx[edge_index]
            edge_weight[torch.isnan(edge_weight)] = 0  

            all_edge_indices.append(edge_index)
            all_edge_weights.append(edge_weight)

        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_weight = torch.cat(all_edge_weights, dim=0)

        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=edge_weight, num_nodes=x.size(0), reduce='mean'
        )
        edge_weight = F.relu(edge_weight)
        edge_weight = F.dropout(
            edge_weight, p=self.drop_percent, training=self.training
        )
        return edge_index, edge_weight

    def get_prompt_weights(self):
        return {
            "feature_prompt": self.feature_prompt_layer.prompt.detach(),
        }

    def forward(self, data):
        x = self.feature_prompt_layer(data.x, data.name)
        x = F.relu(x)
        x = self.shared_token(x)
        edge_index_norm, edge_weight_norm = normalize_edge_index(
            data.edge_index, num_nodes=data.num_nodes, add_self_loops=True
        )
        h = aggregate_features(x, edge_index_norm, edge_weight_norm)
        h = self.balance_token(h, data.name)
        edge_index_re, edge_weight_re = self.multi_reproduce_graph(h, k=self.k, batch=data.batch)
        loop_index = torch.arange(data.num_nodes).unsqueeze(0).expand(2, -1)
        pos_edges = [
            (loop_index, None),
            (edge_index_re.detach(), edge_weight_re.detach()),
        ]

        prelogits_1 = self.sigmod(
            self.gnn_enc(
                x, edge_index_norm, edge_weight=edge_weight_norm, name=data.name, LP=True,
            )
        )
        prelogits_2 = self.sigmod(
            self.gnn_enc(
                x, edge_index_re, edge_weight=edge_weight_re, name=data.name, LP=True,
            )
        )
        loss = self.calc_lower_bound(
            prelogits_1, prelogits_2, pos_edges, temperature=self.temperature
        )
        return loss


class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super(Downstream, self).__init__()
        combinetype = configs.combinetype
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim
        downstream_domain_id = {configs.target_data: 0}

        self.task_name = configs.task_name
        self.num_classes = configs.num_classes
        self.temperature = configs.temperature
        self.k = configs.k
        self.drop_percent = configs.drop_percent

        self.combine_weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.gnn_enc = pretrain_model.gnn_enc
        self.shared_token = pretrain_model.shared_token
        for param in self.gnn_enc.parameters():
            param.requires_grad = False
        for param in self.shared_token.parameters():
            param.requires_grad = False

        prompt_weights = pretrain_model.get_prompt_weights()
        assert (
            "feature_prompt" in prompt_weights
        ), "feature_prompt weights must be provided for MDGPT downstream tasks."
        feature_prompt_weights = prompt_weights["feature_prompt"]
        self.fea_specific_prompts = ComposedPrompt(
            feature_prompt_weights, combinetype=combinetype
        )
        self.specific_prompt = AlignPrompt(
            input_dim, domain_id=downstream_domain_id, combinetype=combinetype
        )
        self.balance_token = AlignPrompt(
            input_dim * 2, domain_id=downstream_domain_id, combinetype=combinetype
        )
        self.alpha = torch.nn.Parameter(
            torch.tensor(0.5)
        )  # merge original adj and reconstructed adj
        class_dim = hidden_dim*2 if self.task_name == 'edge' else hidden_dim
        self.class_prototypes = torch.FloatTensor(self.num_classes, class_dim)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.combine_weight)

    def reproduce_graph(self, x, k, block_size=1024, eps=1e-8):
        edge_index, edge_weight = knn_fast(x, k, block_size, eps=eps)

        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=edge_weight, num_nodes=x.size(0), reduce='mean'
        )
        edge_weight = F.relu(edge_weight)
        edge_weight = F.dropout(
            edge_weight, p=self.drop_percent, training=self.training
        )
        return edge_index, edge_weight

    def embedding_forward(self, data):
        fea_hol_embed = self.specific_prompt(data.x, data.name)
        # meta prompt
        fea_spe_embed = self.fea_specific_prompts(data.x, data.name)
        fea_spe_embed = F.relu(fea_spe_embed)
        fea_spe_embed = self.shared_token(data.x)
        fea_al_embed = self.act(
            self.combine_weight[0][0] * fea_hol_embed
            + self.combine_weight[0][1] * fea_spe_embed
        )

        edge_index_norm, edge_weight_norm = normalize_edge_index(
            data.edge_index, num_nodes=data.num_nodes, add_self_loops=True
        )

        h = aggregate_features(fea_al_embed, edge_index_norm, edge_weight_norm)
        h = self.balance_token(h, data.name)
        edge_index_re, edge_weight_re = self.reproduce_graph(h, k=self.k)

        edge_index_tot, edge_weight_tot = merge_edges(
            edge_index_norm,
            self.alpha * edge_weight_norm,
            edge_index_re,
            (1 - self.alpha) * edge_weight_re,
            num_nodes=data.num_nodes,
        )

        fea_al_embed = self.gnn_enc(
            fea_al_embed,
            edge_index_tot,
            edge_weight=edge_weight_tot,
            prompt_layers=None,
            name=data.name,
            LP=False,
        )

        return fea_al_embed

    def compute_predictions(self, embed, labels=None, is_train=True):
        """
        embed: [num_items, hidden_dim]
        labels: [num_items] only used if is_train=True
        """
        if is_train and labels is not None:
            self.class_prototypes = torch_scatter.scatter(
                src=embed, index=labels, dim=0, reduce="mean", dim_size=self.num_classes
            )  # [num_classes, hidden_dim]

        similarity = cosine_similarity(
            embed, self.class_prototypes, temperature=self.temperature, eps=1e-8
        )  # [num_items, num_classes]

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
        select_edge_embed = torch.cat(
            [node_embed[select_src_idx], node_embed[select_dst_idx]], dim=-1
        )
        return self.compute_predictions(select_edge_embed, edge_type, is_train)

    def graph_classification(self, data, graph_labels, graph_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        graph_embed = torch_scatter.scatter(
            node_embed, data.batch, dim=0, reduce="mean"
        )
        selected_graph_embed = graph_embed[graph_idx]
        return self.compute_predictions(selected_graph_embed, graph_labels, is_train)

    def forward(self, data, labels, idx, is_train=True):
        if self.task_name == "node":
            return self.node_classification(data, labels, idx, is_train)
        elif self.task_name == "edge":
            return self.edge_classification(data, labels, idx, is_train)
        elif self.task_name == "graph":
            return self.graph_classification(data, labels, idx, is_train)
        else:
            raise ValueError("Invalid task name. Use 'node', 'edge', or 'graph'.")


def normalize_edge_index(edge_index, num_nodes, edge_weight=None, add_self_loops=True):
    gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
    edge_index, edge_weight = gcn_norm(
        edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        add_self_loops=add_self_loops,
    )
    return edge_index, edge_weight


def aggregate_features(x, edge_index, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
    # edge_index.flip(0): Reverses PyG's [src, dst] format to [row=dst, col=src].
    # Ensures adj @ x performs correct Source -> Destination message aggregation.
    adj = torch.sparse_coo_tensor(
        edge_index.flip(0), edge_weight, (x.size(0), x.size(0))
    )
    agg = adj @ x
    return torch.cat([x, agg], dim=1)


def merge_edges(edge_index1, edge_weight1, edge_index2, edge_weight2, num_nodes):
    A1 = torch.sparse_coo_tensor(edge_index1, edge_weight1, (num_nodes, num_nodes))
    A2 = torch.sparse_coo_tensor(edge_index2, edge_weight2, (num_nodes, num_nodes))
    A = A1 + A2
    A = A.coalesce()
    return A.indices(), A.values()