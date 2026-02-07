import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils.metapath import auto_generate_metapaths
import torch_geometric.transforms as T
from root import ROOT_DIR
import numpy as np
from torch_geometric.utils import coalesce
import scipy.sparse as sp


class Pretrain(nn.Module):
    def __init__(self, configs, dataset):
        super(Pretrain, self).__init__()

        feature_dim = {}
        for node_type, feat_dim in dataset.num_node_features.items():
            if feat_dim == 0:
                feature_dim[node_type] = dataset[node_type].num_nodes
            else:
                feature_dim[node_type] = feat_dim

        input_dim = configs.input_dim
        if input_dim == -1:
            input_dim = list(feature_dim.values())
        self.in_dim = input_dim  # Should be list per node type
        hidden_dim = configs.hidden_dim

        feat_drop = 0.3
        attn_drop = 0.5
        sample_rate = [7, 7, 1, 1]  # paper, author, subject, term
        tau, lam = configs.temperature, configs.temperature

        self.t_pos = 7
        self.metapaths = auto_generate_metapaths(dataset)
        self.name = list(configs.pretrain_domain_id.keys())[0]

        self.fc_list = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=True)
                                      for input_dim in self.in_dim])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = HeCoMpEncoder(hidden_dim=hidden_dim, attn_drop=attn_drop, num_metapaths=len(self.metapaths), max_keep_edges=-1)
        self.sc = HeCoScEncoder(hidden_dim=hidden_dim, sample_rate=sample_rate, attn_drop=attn_drop, num_neighbor_types=len(self.in_dim))
        self.contrast = ContrastHeCo(hidden_dim, tau, lam)

    def create_mps(self, heterodata):
        dir_path = f'{ROOT_DIR}/datasets/{self.name}/preprocess'
        os.makedirs(dir_path, exist_ok=True)
        mps_path = f'{dir_path}/mps_heco.pt'
        if os.path.exists(mps_path):
            metapaths = torch.load(mps_path)
            return metapaths
        transform = T.AddMetaPaths(metapaths=self.metapaths)
        data = transform(heterodata)
        mps = []
        for edge_type in data.edge_index_dict.keys():
            if 'metapath' in edge_type[1]:  # edge_type = (src, rel, dst)
                mps.append(data.edge_index_dict[edge_type])
        torch.save(mps, mps_path)
        return mps
    
    def create_pos_mask(self, mps, num_nodes, t_pos):
        dir_path = f'{ROOT_DIR}/datasets/{self.name}/preprocess'
        os.makedirs(dir_path, exist_ok=True)
        pos_path = f'{dir_path}/pos_heco_t{t_pos}.pt'

        if os.path.exists(pos_path):
            pos = torch.load(pos_path)
            return pos

        pos = generate_pos_mask(mps, num_nodes, t_pos)
        torch.save(pos, pos_path)
        return pos

    def forward(self, data):
        h_all = []
        feats = list(data.x_dict.values())
        device = feats[0].device
        mps = [mp.to(device) for mp in self.create_mps(data)]
        pos = self.create_pos_mask(mps, feats[0].size(0), t_pos=self.t_pos).to(device)

        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

        z_mp = self.mp(h_all[0], mps, is_training=True)
        z_sc = self.sc(h_all, data)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss
    
    def get_embeds(self, data):
        feats = list(data.x_dict.values())
        device = feats[0].device
        z_mp = F.elu(self.fc_list[0](feats[0]))
        mps = [mp.to(device) for mp in self.create_mps(data)]
        z_mp = self.mp(z_mp, mps, is_training=False)
        return z_mp.detach()
    

class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super(Downstream, self).__init__()
        hidden_dim = configs.hidden_dim
        num_classes = configs.num_classes

        self.task_name = configs.task_name
        self.pretrain_model = pretrain_model
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(hidden_dim, num_classes)
        for m in self.classifier.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def embedding_forward(self, data):
        with torch.no_grad():
            embeds = self.pretrain_model.get_embeds(data)
        return embeds
    
    def compute_predictions(self, embed, labels=None, is_train=True):
        logits = self.classifier(embed) # [num_items, num_classes]
        return logits
        
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


def generate_pos_mask(mps, num_nodes, t_pos):
    """
    Build sparse positive mask (pos) per HeCo Eq.(10).

    Args:
        mps (list[Tensor]): Meta-path edge_index list [edge_index_mp1, ...]
        num_nodes (int): number of nodes
        t_pos (int): threshold T_pos (keep at most T_pos positives per node)

    Returns:
        torch.sparse.FloatTensor: sparse pos mask, shape [N, N]
    """
    print("Generating Positive Mask (pos)...")
    
    # 1) Count C_i(j): how many meta-paths connect node i and j.
    
    edge_index_list = []
    for ei in mps:
        edge_index_list.append(ei)
    
    # concat all edges [2, total_edges]
    cat_edge_index = torch.cat(edge_index_list, dim=1)
    # all weights are 1
    cat_values = torch.ones(cat_edge_index.size(1), device=cat_edge_index.device)

    # coalesce duplicates to accumulate counts -> C_i(j)
    unique_edge_index, counts = coalesce(
        cat_edge_index,
        cat_values,
        num_nodes=num_nodes,
        reduce='sum'
    )

    # 2) Sort & threshold per node (CPU CSR for simplicity)
    
    # to CPU CSR: matrix[i, j] = count
    adj = sp.csr_matrix(
        (counts.cpu().numpy(), (unique_edge_index[0].cpu().numpy(), unique_edge_index[1].cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )
    
    pos_rows = []
    pos_cols = []
    
    # iterate each node
    for i in range(num_nodes):
        # neighbors of node i
        row_start = adj.indptr[i]
        row_end = adj.indptr[i+1]
        
        neighbors = adj.indices[row_start:row_end]
        neighbor_counts = adj.data[row_start:row_end]
        
        if len(neighbors) == 0:
            # isolated node: keep self
            pos_rows.append(i)
            pos_cols.append(i)
            continue
            
        # sort by count desc
        sort_indices = np.argsort(neighbor_counts)[::-1]
        
        # 3) truncate
        if len(neighbors) > t_pos:
            selected_indices = sort_indices[:t_pos]
        else:
            selected_indices = sort_indices # keep all
            
        pos_rows.extend([i] * len(selected_indices))
        pos_cols.extend(neighbors[selected_indices])
        
        if i not in neighbors[selected_indices]:
             pos_rows.append(i)
             pos_cols.append(i)

    # 4) build sparse tensor
    pos_indices = torch.tensor([pos_rows, pos_cols], dtype=torch.long)
    pos_values = torch.ones(len(pos_rows))
    
    # Create SparseTensor (COO format)
    pos_sparse = torch.sparse_coo_tensor(
        pos_indices, 
        pos_values, 
        (num_nodes, num_nodes)
    )
    
    print(f"Pos Mask Completed: {len(pos_rows)}")
    return pos_sparse