import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import numpy as np
import copy


class Pretrain(nn.Module):
    def __init__(self, configs, dataset):
        super().__init__()
        self.num_nodes = dataset.num_nodes
        self.edge_dim = dataset.msg.shape[-1]
        self.hidden_dim = configs.hidden_dim
        self.num_heads = configs.num_heads
        self.time_dim = 128
        self.queue_size = 256
        self.m = 0.999 # Momentum
        self.beta=2.0
        self.tau=0.01
        self.gamma=0.4
        
        # --- INTERNAL SAMPLER INITIALIZATION ---
        # We pass the full dataset here to build the index.
        # This keeps the logic encapsulated within the model.
        self.sampler = TemporalNeighborSampler(dataset)
        
        # Encoders (Query & Key)
        self.gnn_enc = DDGCLTgat(self.hidden_dim, self.edge_dim, self.time_dim, self.hidden_dim, self.num_nodes, num_heads=self.num_heads)
        self.gnn_k = DDGCLTgat(self.hidden_dim, self.edge_dim, self.time_dim, self.hidden_dim, self.num_nodes, num_heads=self.num_heads)
        
        # Initialize Key = Query
        for pq, pk in zip(self.gnn_enc.parameters(), self.gnn_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False
            
        # Dynamic Weight Generators (for Loss)
        self.w_gen_pos = DDGCLDynamicWeightGenerator(self.hidden_dim, self.hidden_dim)
        self.w_gen_neg = DDGCLDynamicWeightGenerator(self.hidden_dim, self.hidden_dim)

        # MoCo Queue buffers
        self.register_buffer("queue", torch.randn(self.queue_size, self.hidden_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_time", torch.zeros(self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update for Key Encoder"""
        for pq, pk in zip(self.gnn_enc.parameters(), self.gnn_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, times):
        """Update Queue with new keys"""
        ptr = int(self.queue_ptr)
        bs = keys.shape[0]
        if ptr + bs <= self.queue_size:
            self.queue[ptr:ptr+bs] = keys
            self.queue_time[ptr:ptr+bs] = times
            self.queue_ptr[0] = (ptr + bs) % self.queue_size
        else:
            self.queue_ptr[0] = 0
            self.queue[:bs] = keys
            self.queue_time[:bs] = times

    def forward(self, batch):
        """
        Forward pass handling Sampling -> Encoding -> Loss.
        input: batch (from TemporalDataLoader)
        """
        # Using the internal sampler to get dual views
        v1_data, v2_data, t_past = self.sampler.get_contrastive_batch(batch.src, batch.t, gamma=self.gamma)
        
        # Handle edge case (isolated nodes)
        if v1_data is None or v2_data is None:
            return torch.tensor(0.0, device=batch.src.device, requires_grad=True)

        device = batch.src.device
        def to_dev(v): return [x.to(device) for x in v] + [len(batch.src)]
        
        v1_args = to_dev(v1_data) # [n_id, edge_index, times, feats, batch_size]
        v2_args = to_dev(v2_data)
        
        t_curr = batch.t.float().to(device)
        t_past = t_past.float().to(device)

        q = self.gnn_enc(*v1_args) # Query
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = self.gnn_k(*v2_args) # Key
            k = F.normalize(k, dim=1)

        loss = debiased_contrastive_loss(
            q, k, t_curr, t_past, 
            self.queue, self.queue_time, 
            self.w_gen_pos, self.w_gen_neg
        )
        return loss, k.detach(), t_past
    

class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super().__init__()
        self.task_name = configs.task_name
        self.num_classes = configs.num_classes
        self.hidden_dim = configs.hidden_dim
        self.num_nodes = dataset.num_nodes
        
        # Register global max time for inference
        self.t_max = float(dataset.t.max().item())

        self.gnn_enc = copy.deepcopy(pretrain_model.gnn_enc)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def embedding_forward(self, data):
        device = next(self.gnn_enc.parameters()).device
        
        n_id = torch.arange(self.num_nodes, device=device)
        mask = data.t <= self.t_max
        
        src = data.src[mask].to(device)
        dst = data.dst[mask].to(device)
        t = data.t[mask].to(device).float()
        msg = data.msg[mask].to(device).float()
        
        edge_index = torch.stack([src, dst], dim=0)
        
        args = [
            n_id,           # Node IDs
            edge_index,     # Full Edge Index
            t,              # Edge Times
            msg,            # Edge Features
            self.num_nodes  # Batch Size (Full Graph)
        ]

        return self.gnn_enc(*args)

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
        

def debiased_contrastive_loss(q, k, t_q, t_k, queue, queue_time, w_gen_pos, w_gen_neg, beta=1.5, tau=0.01):
    """
    Calculates the Debiased Contrastive Loss (J1 - J2).
    """
    # 1. Positive Pair Score: q^T * W_pos * k
    delta_pos = t_q - t_k
    W_pos = w_gen_pos(delta_pos) # [B, D, D]
    wk = torch.bmm(W_pos, k.unsqueeze(-1)).squeeze(-1)
    pos_score = (q * wk).sum(dim=-1, keepdim=True)
    
    # 2. Negative Pair Score: q^T * W_neg * Queue
    delta_neg = t_q.mean() - queue_time # [K]
    W_neg = w_gen_neg(delta_neg) # [K, D, D]
    wq_queue = torch.bmm(W_neg, queue.unsqueeze(-1)).squeeze(-1) # [K, D]
    neg_score = torch.matmul(q, wq_queue.T) # [B, K]
    
    # 3. Debiasing Logic
    l_pos_prob = torch.sigmoid(pos_score)
    loss_pos = F.binary_cross_entropy(l_pos_prob, torch.ones_like(l_pos_prob))
    
    # Reweighting (Hardness) & Bias Correction
    neg_weights = F.softmax(beta * neg_score, dim=1)
    loss_neg_term = -F.softplus(neg_score)
    bias = tau * torch.log(1.0 / (1.0 + torch.exp(pos_score)))
    
    weighted_neg = (neg_weights * loss_neg_term).sum(dim=1, keepdim=True)
    loss_neg = - (1.0 / (1.0 - tau)) * (weighted_neg - bias)
    
    return loss_pos + loss_neg.mean()


# def debiased_contrastive_loss(q, k, t_q, t_k, queue, queue_time, w_gen_pos, w_gen_neg, beta=1.5, tau=0.01):
#     """
#     Inputs from Version 1 (DDGCL context):
#         q, k: [B, D]
#         t_q, t_k: [B]
#         queue: [K, D]
#         queue_time: [K]
#         w_gen_pos, w_gen_neg: Neural Networks for time projection
    
#     Logic from Version 2 (InfoNCE/LogSumExp):
#         L = -pos_sim + logsumexp(neg_sim)
#     """
#     delta_pos = t_q - t_k
#     if delta_pos.dim() == 1: delta_pos = delta_pos.unsqueeze(-1)
    
#     W_pos = w_gen_pos(delta_pos)
#     wk = torch.bmm(W_pos, k.unsqueeze(-1)).squeeze(-1)
    
#     delta_neg = t_q.mean() - queue_time 
#     if delta_neg.dim() == 1: delta_neg = delta_neg.unsqueeze(-1)
    
#     W_neg = w_gen_neg(delta_neg) # [K, D, D]
#     wq_queue = torch.bmm(W_neg, queue.unsqueeze(-1)).squeeze(-1)

#     temperature = 1 / beta
#     anchor = F.normalize(q, p=2, dim=-1)
#     positive = F.normalize(wk, p=2, dim=-1)     
#     negatives = F.normalize(wq_queue, p=2, dim=-1)

#     pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
#     neg_sim = torch.matmul(anchor, negatives.T) / temperature
#     neg_sim_weighted = torch.logsumexp(neg_sim, dim=1)
#     loss = -pos_sim + neg_sim_weighted
    
#     return loss.mean()