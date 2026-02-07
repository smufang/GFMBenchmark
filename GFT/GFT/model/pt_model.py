from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import negative_sampling

EPS = 1e-15


class PretrainModel(nn.Module):
    def __init__(self, encoder, vq, feat_recon_decoder, topo_recon_decoder, topo_sem_recon_decoder):
        super().__init__()

        self.encoder = encoder
        self.vq = vq

        self.feat_recon_decoder = feat_recon_decoder
        self.topo_recon_decoder = topo_recon_decoder
        self.topo_sem_recon_decoder = topo_sem_recon_decoder

        self.sem_encoder = deepcopy(self.encoder)
        self.sem_projector = nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim)

    @property
    def get_encoder(self):
        return self.encoder

    @property
    def get_vq(self):
        return self.vq

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def save_vq(self, path):
        torch.save(self.vq.state_dict(), path)

    def feat_recon(self, z):
        return self.feat_recon_decoder(z)

    def feat_recon_loss(self, z, x, bs=None):
        return F.mse_loss(self.feat_recon(z[:bs]), x[:bs])

    # Reconstructing tree structure, similar to graph reconstruction.
    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):

        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            # Randomly sample positive edges
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1))
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_loss = -torch.log(self.topo_recon_decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.topo_recon_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    # Reconstructing the edge feature between two nodes
    def topo_sem_recon_loss(self, z, edge_index, edge_attr, ratio=1.0):
        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            num_edges = int(edge_index.size(1) * ratio)
            num_edges = max(num_edges, 1)
            perm = torch.randperm(edge_index.size(1))
            perm = perm[:num_edges]
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        loss = F.mse_loss(self.topo_sem_recon_decoder(z), edge_attr)

        return loss

    # Reconstructing the tree representation
    def sem_recon_loss(self, g, quantize, eta=1.0, bs=None):
        orig_x, orig_edge_index, orig_edge_attr = (
            g[0],
            g[1],
            g[2],
        )

        z = self.sem_encoder(orig_x, orig_edge_index, orig_edge_attr).detach()
        h = self.sem_projector(quantize)

        z = F.normalize(z[:bs], dim=-1, p=2)  # N * D
        h = F.normalize(h[:bs], dim=-1, p=2)  # N * D

        loss = (1 - (z * h).sum(dim=-1)).pow_(eta)
        loss = loss.mean()

        return loss

    def ema_update_sem_encoder(self, decay=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.sem_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1 - decay)

    def encode(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr)

    def quantize(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        quantize, indices, commit_loss, _ = self.vq(z)
        return z, quantize, indices, commit_loss

    def forward(self, aug_g, g, topo_recon_ratio=1.0, bs=None, no_codebook=False):
        x, edge_index, edge_attr = aug_g[0], aug_g[1], aug_g[2]
        orig_x, orig_edge_index, orig_edge_attr = g[0], g[1], g[2]

        z, quantize, indices, commit_loss = self.quantize(x, edge_index, edge_attr)
        if no_codebook:
            query = z
            commit_loss = torch.tensor(0.0, device=z.device)
        else:
            query = quantize

        feat_recon_loss = self.feat_recon_loss(query, orig_x, bs=bs)
        topo_recon_loss = self.topo_recon_loss(query, orig_edge_index, ratio=topo_recon_ratio)
        topo_sem_recon_loss = self.topo_sem_recon_loss(query, orig_edge_index, orig_edge_attr, ratio=topo_recon_ratio)
        sem_recon_loss = self.sem_recon_loss(g, query, eta=1.0, bs=bs)

        losses = {
            'feat_recon_loss': feat_recon_loss,
            'topo_recon_loss': topo_recon_loss,
            'topo_sem_recon_loss': topo_sem_recon_loss,
            'sem_recon_loss': sem_recon_loss,
            'commit_loss': commit_loss,
        }

        return z, quantize, indices, losses
