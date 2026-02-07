import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch_geometric.nn.models.tgn import LastNeighborLoader


class Pretrain(nn.Module):
    """
    This model does not support pretraining.
    """
    def __init__(self, configs, dataset):
        super().__init__()
        # TGN configs
        self.num_nodes = dataset.num_nodes
        self.raw_msg_dim = dataset.msg.shape[-1]   # dimension of raw message
        self.memory_dim = 172
        self.time_dim = 100
        self.hidden_dim = configs.hidden_dim
        self.batch_size = configs.batch_size
        
        # Register as buffers so they move with .to(device)
        self.register_buffer('msg', dataset.msg)
        self.register_buffer('t', dataset.t)

        self.gnn_enc = TGNLayers(
            num_nodes=self.num_nodes,
            raw_msg_dim=self.raw_msg_dim,
            memory_dim=self.memory_dim,
            time_dim=self.time_dim,
            embedding_dim=self.hidden_dim,
            heads=configs.num_heads,
            dropout=configs.dropout,
        )

        self.LpPredictor = LpTGN(self.hidden_dim)
        self.neighbor_loader = LastNeighborLoader(dataset.num_nodes, size=10)
        self.criterion = nn.BCEWithLogitsLoss()

        self.need_restart = True

    def restart(self):
        if self.need_restart:
            self.neighbor_loader.reset_state()
            self.gnn_enc.reset_memory()
            self.need_restart = False

    def update(self, batch):
        self.gnn_enc.detach_memory()
        self.gnn_enc.update_memory(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbor_loader.insert(batch.src.to('cpu'), batch.dst.to('cpu'))

    def get_embeddings(self, batch):
        device = batch.src.device
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id.cpu())
        n_id = n_id.to(device)
        edge_index = edge_index.to(device)
        e_id = e_id.to(device)

        assoc = torch.empty(self.num_nodes, dtype=torch.long, device=n_id.device)
        assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        z = self.gnn_enc(n_id, edge_index, e_id, self.t, self.msg)
        return z, assoc
    
    def forward(self, batch):
        self.restart()
        z, assoc = self.get_embeddings(batch)
        pos_out = self.LpPredictor(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = self.LpPredictor(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        loss = self.criterion(pos_out, torch.ones_like(pos_out)) + self.criterion(neg_out, torch.zeros_like(neg_out))
        return loss


class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super().__init__()
        self.task_name = configs.task_name       # 'node'
        self.num_classes = configs.num_classes
        self.hidden_dim = configs.hidden_dim
        self.num_nodes = dataset.num_nodes

        self.gnn_enc = pretrain_model.gnn_enc
        for param in self.gnn_enc.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    @torch.no_grad()
    def embedding_forward(self, data):
        self.gnn_enc.eval()
        if 'name' in data:
            del data.name
        device = data.src.device
        all_nodes = torch.arange(self.num_nodes).to(device)
        edge_index = data.edge_index.to(device)
        e_id = data.e_id.to(device)
        t = data.t.to(device)
        msg = data.msg.to(device)
        z = self.gnn_enc(all_nodes, edge_index, e_id, t, msg)
        return z

    def compute_predictions(self, embed, labels=None, is_train=True):
        """
        embed: [num_items, hidden_dim]
        labels: [num_items] only used if is_train=True
        """
        logits = self.classifier(embed)  # [num_items, num_classes]
        if is_train:
            return logits
        else:
            return F.softmax(logits, dim=1)

    def node_classification(self, data, labels, node_idx, is_train=True):
        embed = self.embedding_forward(data)[node_idx]
        return self.compute_predictions(embed, labels, is_train)

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