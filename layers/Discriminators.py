import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorBilinear(nn.Module):
    def __init__(self, hidden_dim):
        super(DiscriminatorBilinear, self).__init__()
        self.bilin = nn.Bilinear(hidden_dim, hidden_dim, out_features=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, global_emb, pos_node_emb, neg_node_emb, pos_bias=None, neg_bias=None):
        # global_emb: [hidden_dim]
        # pos_node_emb, neg_node_emb: [num_nodes, hidden_dim]
        global_emb = global_emb.unsqueeze(0).expand_as(pos_node_emb)  # [num_nodes, hidden_dim]

        pos_scores = self.bilin(pos_node_emb, global_emb).squeeze(-1)
        neg_scores = self.bilin(neg_node_emb, global_emb).squeeze(-1)

        if pos_bias is not None:
            pos_scores += pos_bias
        if neg_bias is not None:
            neg_scores += neg_bias

        return torch.cat((pos_scores, neg_scores), 0)


class DiscriminatorCos(nn.Module):
    def __init__(self):
        super(DiscriminatorCos, self).__init__()

    def forward(self, global_emb, pos_node_emb, neg_node_emb, pos_bias=None, neg_bias=None):
        # global_emb: [hidden_dim]
        # pos_node_emb, neg_node_emb: [num_nodes, hidden_dim]
        global_emb = global_emb.unsqueeze(0).expand_as(pos_node_emb)  # [num_nodes, hidden_dim]

        pos_scores = F.cosine_similarity(pos_node_emb, global_emb, dim=-1)  # [num_nodes]
        neg_scores = F.cosine_similarity(neg_node_emb, global_emb, dim=-1)  # [num_nodes]

        if pos_bias is not None:
            pos_scores += pos_bias
        if neg_bias is not None:
            neg_scores += neg_bias

        return torch.cat((pos_scores, neg_scores), dim=0)
