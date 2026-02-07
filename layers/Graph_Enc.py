import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GATConv
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from .Conv import SimpleHGNConv, HeCoGCNConv
from .Attention import HeCoInterAttn, HeCoIntraAttn, GraphAttentionEmbedding


class GCNLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, activation="relu", normalize=True, add_self_loops=True):
        super(GCNLayers, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "elu":
            self.act_fn = F.elu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "prelu":
            self.act_fn = None  # PReLU needs separate handling
        elif activation == "none":
            self.act_fn = nn.Identity()
        else:
            self.act_fn = F.relu  # default

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim, 
                                      add_self_loops=add_self_loops,
                                      normalize=normalize))

            if activation == "prelu":
                self.acts.append(nn.PReLU())
            else:
                self.acts.append(nn.Identity())  # Use Identity as placeholder for functional activations

            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    m.lin.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None, LP=False, prompt_layers=None, name=None):
        """
        Forward pass through stacked GCN layers.

        Args:
            x: [num_nodes/total_nodes_in_batch, input_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges], optional edge weights
            LP (bool): link prediction
            prompt_layers (list[nn.Module], optional): Prompt modules to inject per layer.
            name (list(str)): Optional input for prompt layers. [num_nodes/total_nodes_in_batch]

        Returns:
            x: [num_nodes/total_nodes_in_batch, hidden_dim].
        """

        if prompt_layers:
            assert len(prompt_layers) == self.num_layers

        h = x
        for i in range(self.num_layers):
            out = self.convs[i](h, edge_index, edge_weight=edge_weight)
            if self.act_fn is not None:
                out = self.act_fn(out)  # Functional activation (ReLU, GELU, etc.)
            else:
                out = self.acts[i](out)  # Module activation (PReLU)
            # out = self.dropout(out)

            # Residual connection (skip first layer)
            h = h + out if i > 0 else out

            if prompt_layers:
                h = prompt_layers[i](h, name)

            if LP:
                h = self.bns[i](h)
                h = self.dropout(h)

        return h


class GATLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads=2, concat=True, bias=True, dropout=0.1,
                 activation="relu"):
        super(GATLayers, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "elu":
            self.act_fn = F.elu
        elif activation == "prelu":
            self.act_fn = None  # PReLU needs separate handling
        elif activation == "none":
            self.act_fn = nn.Identity()
        else:
            self.act_fn = F.relu  # default

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if concat is True:
                if hidden_dim % num_heads != 0:
                    raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
                self.convs.append(GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, concat=True, bias=bias))
            else:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=num_heads, concat=False, bias=bias))

            if activation == "prelu":
                self.acts.append(nn.PReLU())
            else:
                self.acts.append(nn.Identity())  # Use Identity as placeholder for functional activations

            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                nn.init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    m.lin.bias.data.fill_(0.0)

    def forward(self, x, edge_index, LP=False, prompt_layers=None, name=None):
        """
        Forward pass through stacked GCN layers.

        Args:
            x: [num_nodes/total_nodes_in_batch, input_dim]
            edge_index: [2, num_edges]
            LP (bool): link prediction
            prompt_layers (list[nn.Module], optional): Prompt modules to inject per layer.
            name (list(str)): Optional input for prompt layers. [num_nodes/total_nodes_in_batch]

        Returns:
            x: [num_nodes/total_nodes_in_batch, hidden_dim].
        """

        if prompt_layers:
            assert len(prompt_layers) == self.num_layers

        h = x
        for i in range(self.num_layers):
            out = self.convs[i](h, edge_index)
            if self.act_fn is not None:
                out = self.act_fn(out)  # Functional activation (ReLU, GELU, etc.)
            else:
                out = self.acts[i](out)  # Module activation (PReLU)
            # out = self.dropout(out)

            # Residual connection (skip first layer)
            h = h + out if i > 0 else out

            if prompt_layers:
                h = prompt_layers[i](h, name)

            if LP:
                h = self.bns[i](h)
                h = self.dropout(h)

        return h


class SimpleHGNLayers(nn.Module):
    """
    PyG implementation fully aligned with DGL 'myGAT' class.
    Structure:
      1. Node Type Projection (Linear + Bias)
      2. Input GAT Layer (No Residual)
      3. Hidden GAT Layers (With Residual)
      4. Output GAT Layer (Project to num_classes, Mean over heads)
      5. L2 Normalization
    """
    def __init__(self,
                 in_dims,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 num_heads,
                 num_etypes,
                 edge_dim,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 activation='elu',
                 negative_slope=0.2,
                 residual=True,
                 alpha=0.05):
        super(SimpleHGNLayers, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "elu":
            self.act_fn = F.elu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "none":
            self.act_fn = nn.Identity()
        else:
            self.act_fn = F.relu  # default

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.hidden_dim, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layers = nn.ModuleList()
        # Layer 0: Input Layer
        # residual=False, activation=self.activation
        self.layers.append(
            SimpleHGNConv(
                self.hidden_dim, self.hidden_dim, num_heads[0], num_etypes, edge_dim, 
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=False,
                activation=self.act_fn,
                alpha=alpha
            )
        )

        # Layers 1 to num_layers-1: Hidden Layers
        # DGL: residual=residual, activation=self.activation
        for l in range(1, num_layers):
            self.layers.append(
                SimpleHGNConv(
                    self.hidden_dim * num_heads[l-1], self.hidden_dim, num_heads[l], num_etypes, edge_dim,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=self.act_fn,
                    alpha=alpha
                )
            )
            
        # Output Layer:
        # DGL: residual=residual, activation=None
        self.layers.append(
            SimpleHGNConv(
                self.hidden_dim * num_heads[-2], num_classes, num_heads[-1], num_etypes, edge_dim,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=None,
                alpha=alpha
            )
        )

        # Epsilon for L2 normalization (aligned with DGL)
        self.register_buffer('epsilon', torch.FloatTensor([1e-12]))

    def forward(self, x, edge_index, edge_type, node_type):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_type: [E]
        node_type: [N]
        """
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)  
        # h: [N, hidden_dim]

        # Convert sparse tensor to dense if necessary
        if x.is_sparse:
            x = x.to_dense()

        unique_types = node_type.unique()
        for ntype in unique_types:
            mask = (node_type == ntype)                 # [N]
            feat = x[mask]                             # [N_t, in_dim_t]
            h[mask] = self.fc_list[ntype](feat)  # [N_t, hidden_dim]

        res_attn = None
        for l in range(self.num_layers):
            # h: [N, hidden_dim * heads[l-1]] (l>0)
            # or [N, hidden_dim] (l=0)
            h, res_attn = self.layers[l](h, edge_index, edge_type, res_attn=res_attn) # [N, heads[l], hidden_dim]
            h = h.flatten(1) # [N, heads[l] * hidden_dim]

        logits, _ = self.layers[-1](h, edge_index, edge_type, res_attn=None) # [N, heads_last, num_classes]

        logits = logits.mean(1) # [N, num_classes]
        norm = torch.norm(logits, dim=1, keepdim=True)  # [N, 1]
        logits = logits / torch.max(norm, self.epsilon) # [N, num_classes]

        return logits


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1, activation="relu"):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        if not hidden_dim:
            hidden_dim = 4 * output_dim

        # Feed Forward Networks
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1),
                                 self.act,
                                 self.dropout,
                                 nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1),
                                 self.dropout)

    def forward(self, x, reduce_mean=False):
        out = self.mlp(x)
        if reduce_mean:
            out = torch.mean(out, dim=0)
        return out
    

class HeCoScEncoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, attn_drop, num_neighbor_types):
        super(HeCoScEncoder, self).__init__()
        self.sample_rate = sample_rate # [num_neighbor_types]
        self.intra_layers = nn.ModuleList([
            HeCoIntraAttn(hidden_dim, attn_drop) for _ in range(num_neighbor_types)
        ])
        
        self.inter_att = HeCoInterAttn(hidden_dim, attn_drop)

    def forward(self, h_all, heterodata):
        target_type = heterodata.node_types[0]
        h_target = h_all[0] # [N_target, d_target]
        device = h_target.device
        edge_types = heterodata.edge_types
        type_to_idx = {ntype: idx for idx, ntype in enumerate(heterodata.node_types)}

        embeds = []
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            if dst_type != target_type:
                continue
            
            src_node_idx = type_to_idx[src_type]
            h_nei = h_all[src_node_idx] # [N_source, d_source]
            edge_index = heterodata[edge_type].edge_index  # [2, E], src->dst

            src_idx, tgt_idx = edge_index
            nei_lists = [[] for _ in range(h_target.size(0))]  # len N_target
            for s_idx, t_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
                nei_lists[t_idx].append(s_idx)

            sample_num = int(self.sample_rate[src_node_idx])
            edge_index_sampled = torch.empty(
                2, h_target.size(0) * sample_num, dtype=torch.long, device=device
            )  # [2, N_target*sample_num]

            for t_idx, per_node_nei in enumerate(nei_lists):
                if len(per_node_nei) == 0:
                    # fallback: keep valid index within source range
                    safe_idx = min(t_idx, h_nei.size(0) - 1)
                    chosen = torch.full((sample_num,), safe_idx, dtype=torch.long, device=device)
                else:
                    candidates = torch.as_tensor(per_node_nei, dtype=torch.long, device=device)
                    if len(per_node_nei) >= sample_num:
                        perm = torch.randperm(len(per_node_nei), device=device)[:sample_num]
                        chosen = candidates[perm]
                    else:
                        replace_idx = torch.randint(len(per_node_nei), (sample_num,), device=device)
                        chosen = candidates[replace_idx]
                target_index = torch.full((sample_num,), t_idx, dtype=torch.long, device=device)
                start = t_idx * sample_num
                end = start + sample_num
                edge_index_sampled[0, start:end] = chosen
                edge_index_sampled[1, start:end] = target_index
            
            h = self.intra_layers[src_node_idx](x=h_nei, edge_index=edge_index_sampled, x_refer=h_target)
            embeds.append(F.elu(h))

        z_mc = self.inter_att(embeds)
        return z_mc
        


class HeCoMpEncoder(nn.Module):
    def __init__(self, hidden_dim, attn_drop, num_metapaths, max_keep_edges=100000):
        super(HeCoMpEncoder, self).__init__()
        self.num_metapaths = num_metapaths
        self.gcn_layers = nn.ModuleList([
            HeCoGCNConv(hidden_dim, hidden_dim) for _ in range(num_metapaths)
        ])
        self.att = HeCoInterAttn(hidden_dim, attn_drop)
        self.max_keep_edges = max_keep_edges

    def forward(self, x, mp_edge_index, is_training=True):
        """
        Args:
            x: [N, hidden_dim]
            mp_edge_index: list([2, E]*num_metapaths)
        """
        embeds = []
        for i in range(self.num_metapaths):
            edge_index = mp_edge_index[i]
            if is_training and self.max_keep_edges > 0:
                num_edges = edge_index.size(1)
                if num_edges > self.max_keep_edges:
                    # Randomly sample a subset of edges to avoid too many edges leading to the memory issue
                    perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_keep_edges]
                    edge_index = edge_index[:, perm]
            out = self.gcn_layers[i](x, edge_index)
            embeds.append(out)
        z_mp = self.att(embeds)
        return z_mp
    

class TGNLayers(nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, heads=2, dropout=0.1):
        """
        A unified TGN module that handles Memory, GNN Encoding, and Decoding.
        
        Args:
            num_nodes: Total number of nodes in the graph (for Memory initialization).
            raw_msg_dim: Dimension of the raw edge features (data.msg).
            memory_dim: Dimension of the hidden memory state.
            time_dim: Dimension of the time encoding.
            embedding_dim: Dimension of the final node embedding.
            heads: Number of attention heads in the GNN encoder.
            dropout: Dropout rate in the GNN encoder.
        """
        super().__init__()        
        # Stores the state of all nodes. Updated after every batch.
        self.memory = TGNMemory(
            num_nodes,
            raw_msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        
        # Aggregates information from the spatial-temporal neighborhood.
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc,
            heads=heads,
            dropout=dropout
        )

    def forward(self, n_id, edge_index, e_id, t, msg):
        """
        Forward pass to compute embeddings and predictions.

        Args:
            n_id: Global IDs of all nodes involved in computation (from neighbor_loader).
            edge_index: Adjacency list of the sampled subgraph.
            e_id: Global IDs of the edges in the sampled subgraph.
            t: Global tensor containing timestamps for all events (data.t).
            msg: Global tensor containing features for all events (data.msg).
        Returns:
            z: Node embeddings
        """
        
        # Get the current memory state `z` and the last update time for nodes in `n_id`
        z, last_update = self.memory(n_id)
        # Update embeddings using the temporal subgraph structure
        z = self.gnn(
            z, 
            last_update, 
            edge_index, 
            t[e_id], 
            msg[e_id]
        ) # [num_nodes_in_subgraph, embedding_dim]
        return z

    def update_memory(self, src, dst, t, msg):
        """
        Updates the memory state of the nodes.
        Must be called after the loss calculation and before the next batch.
        """
        self.memory.update_state(src, dst, t, msg)

    def reset_memory(self):
        """Resets the memory to the initial state (zeros)."""
        self.memory.reset_state()

    def detach_memory(self):
        """
        Detaches the memory from the computation graph.
        Crucial for Truncated Backpropagation Through Time (TBPTT) to save GPU memory.
        """
        self.memory.detach()


class DDGCLTgat(nn.Module):
    """
    Time-aware Graph Encoder based on TransformerConv.
    Corresponds to 'Base_encoder' + 'Attention_Net' in original TF code.
    """
    def __init__(self, node_dim, edge_dim, time_dim, out_dim, num_nodes, num_heads=8):
        super().__init__()
        self.time_enc = nn.Linear(1, time_dim)
        # Assuming node IDs are 0 to num_nodes-1
        self.node_emb = nn.Embedding(num_nodes, node_dim)
        
        # Input edge dim = raw edge feature + time encoding
        input_edge_dim = edge_dim + time_dim 

        self.conv1 = TransformerConv(node_dim, out_dim, heads=num_heads, edge_dim=input_edge_dim)
        self.conv2 = TransformerConv(out_dim*num_heads, out_dim, heads=num_heads, edge_dim=input_edge_dim)
        self.projector = nn.Linear(out_dim*num_heads, out_dim)

    def forward(self, n_id, edge_index, edge_times, edge_feats, batch_size):
        x = self.node_emb(n_id)
        t_emb = torch.cos(self.time_enc(edge_times.unsqueeze(-1)))
        edge_attr = torch.cat([edge_feats, t_emb], dim=-1)
        
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return self.projector(x[:batch_size])


class DDGCLDynamicWeightGenerator(nn.Module):
    """
    Generates dynamic weight matrix W based on time delta.
    Corresponds to the dense layers generating W_pos/W_neg in TF code.
    """
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.time_enc = nn.Linear(1, time_dim)
        self.w_gen = nn.Sequential(
            nn.Linear(time_dim, out_dim * out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim

    def forward(self, delta_t):
        t_emb = torch.cos(self.time_enc(delta_t.unsqueeze(-1)))
        W = self.w_gen(t_emb).reshape(-1, self.out_dim, self.out_dim)
        return W


if __name__ == '__main__':
    from data_provider.data_generator import *

    dataset = {'ACM': acm,
               'DBLP': dblp,
               'BZR': bzr,
               'COX2': cox2}
    from data_provider.data_loader import pretrain_loader

    dataloader = pretrain_loader(dataset, batch_size=1)

    with torch.no_grad():
        for batch in dataloader:
            model1 = GATLayers(input_dim=batch.x.size(1), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1,
                               concat=False,
                               activation="relu")
            model2 = GATLayers(input_dim=batch.x.size(1), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1,
                               concat=True,
                               activation="relu")
            model1.eval()
            print(batch)
            x = model1(batch.x, batch.edge_index, name=batch.name, LP=True)
            print(f"Output shape: {x.shape}")
            x = model2(batch.x, batch.edge_index, name=batch.name, LP=True)
            print(f"Output shape: {x.shape}")
            break
