import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import TransformerConv


class HeCoInterAttn(nn.Module):
    """
    Fuses embeddings from different node types (e.g., Paper-Author, Paper-Subject)
    into one final representation.
    """
    def __init__(self, hidden_dim, attn_drop):
        super(HeCoInterAttn, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x: x

    def forward(self, embeds):
        # embeds: List of Tensors, each [N, hidden_dim]
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            # Global importance score for the node type
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        #print("Schema-view type weights (beta):", beta.data.cpu().numpy())
        
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class HeCoIntraAttn(MessagePassing):
    """
    Aggregates neighbors of a specific type using attention.
    Replaces manual F.embedding and neighbor selection.
    """
    def __init__(self, hidden_dim, attn_drop):
        super(HeCoIntraAttn, self).__init__(aggr='add') # Weighted sum aggregation
        self.att = nn.Parameter(torch.empty(size=(1, 2 * hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x: x
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, edge_index, x_refer):
        # x: Neighbor node features, x_refer: Target node features
        # edge_index: [2, E]
        return self.propagate(edge_index, x=x, x_refer=x_refer)
    
    def message(self, x_j, x_refer_i, index, ptr, size_i):
        all_emb = torch.cat([x_refer_i, x_j], dim=-1)
        
        attn_curr = self.attn_drop(self.att)
        alpha = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        alpha = softmax(alpha, index, ptr, size_i) 
        
        return x_j * alpha


class GraphAttentionEmbedding(nn.Module):
    """
    The GNN Encoder part of TGN.
    It combines node Memory, Neighbor features, and Time encodings to generate embeddings.
    """
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, heads=2, dropout=0.1):
        super().__init__()
        self.time_enc = time_enc
        # Edge dimension = raw message dimension + time encoding dimension
        edge_dim = msg_dim + time_enc.out_channels
        
        # We use TransformerConv (GAT-like) to aggregate neighbor information
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=heads,
                                    dropout=dropout, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        """
        Args:
            x: Current memory of nodes [num_nodes, memory_dim]
            last_update: Timestamp of the last update for each node [num_nodes]
            edge_index: Graph connectivity (adjacency list) from the subgraph sampler
            t: Timestamps of the edges in the subgraph
            msg: Raw features/messages of the edges in the subgraph
        """
        # Compute relative time elapsed since the last update
        rel_t = last_update[edge_index[0]] - t
        # Encode the relative time
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        # Concatenate time encoding and raw message features to form edge attributes
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        # Apply Graph Convolution
        return self.conv(x, edge_index, edge_attr)