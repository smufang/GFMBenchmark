import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.aug import DataAugment
from layers import *
from typing import Union
from collections import defaultdict


class LpTGN(torch.nn.Module):
    def __init__(self, in_channels):
        super(LpTGN, self).__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class LpGPT(nn.Module):
    """Link prediction task in MDGPT, SAMGPT"""

    def __init__(self):
        super(LpGPT, self).__init__()
        self.sigmod = nn.ELU()

    @staticmethod
    def prompt_pretrain_sample(edge_index: torch.Tensor, name_ids, num_nodes, n=50):
        """
        For each node, sample 1 positive edge and n negative edges.
        """
        original_device = edge_index.device
        device = torch.device("cpu")
        edge_index = edge_index.cpu()
        edge_index = edge_index.cpu()
        if isinstance(name_ids, torch.Tensor):
            name_ids = name_ids.cpu()

        name_to_nodes = defaultdict(list)
        for node_idx, name in enumerate(name_ids):
            name_to_nodes[name].append(node_idx)

        # Build adjacency list
        adj_lists = [[] for _ in range(num_nodes)]
        src_nodes, dst_nodes = edge_index
        for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
            adj_lists[src].append(dst)

        sample_edge_index = torch.zeros((num_nodes, n + 1), dtype=torch.long, device=device)

        for i in range(num_nodes):
            neighbors = adj_lists[i]
            # Positive sample
            if neighbors:
                pos_dst = neighbors[torch.randint(len(neighbors), (1,)).item()]
            else:
                pos_dst = i
            sample_edge_index[i, 0] = pos_dst

            # Negative samples
            all_neighbors = set(neighbors + [i])
            # Ensure we sample from nodes that are not neighbors and from the same domain
            current_name = name_ids[i]
            same_name_nodes = set(name_to_nodes[current_name])
            neg_candidates = list(same_name_nodes - all_neighbors)
            if len(neg_candidates) >= n:
                neg_dst = torch.tensor(neg_candidates, device=device)[torch.randperm(len(neg_candidates))[:n]]
            else:
                neg_dst = torch.tensor(neg_candidates, device=device)[
                    torch.randint(len(neg_candidates), (n,), device=device)]

            sample_edge_index[i, 1:] = neg_dst

        return sample_edge_index.to(original_device)

    @staticmethod
    def compare_loss(features, samples, temperature=1.0):
        """
        features: [num_nodes, feat_dim]
        samples: [num_nodes, (1 + n)] the first is positive sample, others are negative samples
        """
        sim_matrix = F.cosine_similarity(
            features.unsqueeze(1),  # [num_nodes, 1, feat_dim]
            features[samples],  # [num_nodes, 1+n, feat_dim]
            dim=2  # [num_nodes, 1+n]
        )
        exp_sim = torch.exp(sim_matrix / temperature)
        pos = exp_sim[:, 0]
        neg = exp_sim[:, 1:].sum(dim=1)
        loss = -torch.log(pos / (neg + 1e-8))
        return loss.mean()

    def forward(self, gnn: Union[GATLayers, GCNLayers], data: Data, prompt_layers=None):
        h = gnn(data.x, data.edge_index, prompt_layers=prompt_layers, name=data.name, LP=True)
        return self.sigmod(h)


class GraphCLGPT(nn.Module):
    """Graph contrastive learning task in MDGPT, SAMGPT"""

    def __init__(self, hidden_dim, sim='bil'):
        super(GraphCLGPT, self).__init__()
        self.sigmoid = nn.Sigmoid()
        if sim == 'bil':
            self.disc = DiscriminatorBilinear(hidden_dim)
        elif sim == 'cos':
            self.disc = DiscriminatorCos()
        else:
            raise ValueError()

    def forward(self, gnn: Union[GATLayers, GCNLayers], data: Data, samp_bias1=None, samp_bias2=None,
                aug_type='edge', drop_percent=0.1, prompt_layers=None):

        device = data.x.device
        aug_data = DataAugment(data.cpu(), drop_percent).build_aug(aug_type).to(device)
        data = data.to(device)

        h_0 = gnn(data.x, data.edge_index)  # [num_nodes, hidden_dim]
        h_2 = gnn(aug_data.shuffled_x, data.edge_index, prompt_layers=prompt_layers, name=data.name)

        if aug_type == 'edge':

            h_1 = gnn(data.x, aug_data.edge_index_aug1, prompt_layers=prompt_layers, name=data.name)
            h_3 = gnn(data.x, aug_data.edge_index_aug2, prompt_layers=prompt_layers, name=data.name)

        elif aug_type == 'mask':

            h_1 = gnn(aug_data.aug_feature1, data.edge_index, prompt_layers=prompt_layers, name=data.name)
            h_3 = gnn(aug_data.aug_feature2, data.edge_index, prompt_layers=prompt_layers, name=data.name)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gnn(aug_data.aug_feature1, aug_data.edge_index_aug1, prompt_layers=prompt_layers,
                      name=aug_data.name_aug1)
            h_3 = gnn(aug_data.aug_feature2, aug_data.edge_index_aug2, prompt_layers=prompt_layers,
                      name=aug_data.name_aug2)
        else:
            assert False

        c_1 = torch.mean(h_1, 0)  # [hidden_dim]
        c_1 = self.sigmoid(c_1)

        c_3 = torch.mean(h_3, 0)
        c_3 = self.sigmoid(c_3)

        logits_1 = self.disc(c_1, h_0, h_2, pos_bias=samp_bias1, neg_bias=samp_bias2)
        logits_2 = self.disc(c_3, h_0, h_2, pos_bias=samp_bias1, neg_bias=samp_bias2)

        logits = logits_1 + logits_2
        return logits, aug_data.lbl


class ContrastHeCo(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(ContrastHeCo, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc
    

if __name__ == '__main__':
    from data_provider.data_generator import *

    dataset = {'ACM': acm,
               'DBLP': dblp,
               'BZR': bzr,
               'COX2': cox2}
    from data_provider.data_loader import pretrain_loader

    dataloader = pretrain_loader(dataset, batch_size=2)
    hidden_dim = 64
    with torch.no_grad():
        for data in dataloader:
            # model = LpGPT()
            # gnn = GATLayers(input_dim=data.x.size(1), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1,
            #                 concat=True,
            #                 activation="prelu")
            # lg = model(gnn, data)
            sample_edge_index = LpGPT.prompt_pretrain_sample(data.edge_index, data.name, data.x.size(0),
                                                             n=50)
            print(sample_edge_index)
