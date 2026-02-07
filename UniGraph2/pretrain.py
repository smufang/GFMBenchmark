import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional
import random
from torch_geometric.data import Data
from models.unigraph2_model import UniGraph2Model
from trainers.trainer import UniGraph2Trainer
from models.unigraph2 import UniGraph2
from data.lp_dataset import LinkPredictionDataset
from data.nc_dataset import NodeClassificationDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData, TemporalData, Data, Batch
from utils.utils import create_x
from utils.compress_func import compress_pca
from utils import compress_func
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils.subbatch_loader import BatchGraphLoader
from torch_sparse import SparseTensor
from collections import deque
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--input_dim', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=39, help='seed')
    parser.add_argument('--compress_function', type=str, default='pca',
                            help='dimension alignment method: pca/svd')
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(args)
    print('-' * 10)

    return vars(args)

pretrain_dict_exp4 = {
    'Photo': photo,
    'Computers': computers,
    'COX2': cox2,
    'PROTEINS': proteins,
    'ENZYMES': enzymes,
    'Elliptic': elliptic
}

pretrain_dict_exp3 = {
    'Cora': cora,
    'ACM': acm,
    'DBLP': dblp
}

pretrain_dict_exp1 = {
    'Cora': cora,
    'ogbn-arxiv': ogbn_arxiv,
    'ACM': acm,
    'DBLP': dblp,
    'Reddit': reddit,
    'Texas': texas,
    'Wisconsin': wisconsin,
    'Cornell': cornell,
    'IMDB': imdb,
    'Photo': photo,
    'Computers': computers,
    'Amazon': amazon,
    'Amazon-HeTGB': amazonh,
    'HIV': hiv,
    'COX2': cox2,
    'PROTEINS': proteins,
    'ENZYMES': enzymes,
    'FB15K-237': fb15k237,
    'NELL': nell,
    'Elliptic': elliptic
}

PRETRAIN_DICT = {
    1: pretrain_dict_exp1,
    2: pretrain_dict_exp1,
    3: pretrain_dict_exp3,
    4: pretrain_dict_exp4,
}

def multidata_sampler(data: Data, name, batch_size=16, num_workers=0):
    # This method is for multiple small graphs been merged into a single large graph using Batch.from_data_list()
    data = create_x(data)
    x = data.x if not (data.x.layout == torch.sparse_csr or data.x.layout == torch.sparse_coo
                       ) else data.x.to_dense()  # Ensure x is dense for NeighborLoader

    loader = BatchGraphLoader(
        data,
        graph_label_index=None,
        graph_label=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    for subgraph in loader:
        subgraph.edge_type = torch.zeros(subgraph.num_edges, dtype=torch.long)
        subgraph.edge_attr = torch.zeros((subgraph.num_edges, 1), dtype=torch.float)
        subgraph.raw_texts = [f"node {i}" for i in range(subgraph.num_nodes)]
        subgraph.relation_texts = ['to']
        subgraph.name = name
        del subgraph.batch
        del subgraph.graph_label_index
        yield subgraph


def data_sampler(data: Data, name, num_neighbors=(10, 5), batch_size=16, num_workers=0):
    data = create_x(data)
    if isinstance(data.x, SparseTensor):
        x = data.x.to_dense()
    else:
        if hasattr(data.x, "layout") and (data.x.layout == torch.sparse_csr or data.x.layout == torch.sparse_coo):
            x = data.x.to_dense()
        else:
            x = data.x

    raw_texts = data.raw_texts if hasattr(data, 'raw_texts') else None
    edge_type = data.edge_type if hasattr(data, 'edge_type') else None
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    relation_texts = data.relation_texts if hasattr(data, 'relation_texts') else None

    # control the input nodes for big graphs
    if hasattr(data, 'batch') and data.batch is not None:
        input_nodes = []
        batch_ids = data.batch.unique()
        for b in batch_ids:
            node_idx = (data.batch == b).nonzero(as_tuple=True)[0]
            num_sample = len(node_idx) // 16 + 1
            sampled = node_idx[torch.randperm(len(node_idx))[:num_sample]]
            input_nodes.extend(sampled.tolist())
        input_nodes = torch.tensor(input_nodes)
    else:
        num_sample = data.num_nodes ** 2 // data.num_edges + 1
        sampled = torch.randperm(data.num_nodes)[:num_sample]
        input_nodes = sampled

    loader = NeighborLoader(
        Data(x=x, edge_index=data.edge_index, edge_type=edge_type, edge_attr=edge_attr),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=True,
        replace=False,
        directed=False,
        num_workers=num_workers,
        persistent_workers=False
    )

    for subgraph in loader:
        out_subgraph = Data(x=subgraph.x, edge_index=subgraph.edge_index)
        node_ids = subgraph.n_id.tolist()
        edge_ids = subgraph.e_id.tolist() if hasattr(subgraph, 'e_id') else []

        if edge_type is None:
            out_subgraph.edge_type = torch.zeros(len(edge_ids), dtype=torch.long)
        else:
            out_subgraph.edge_type = subgraph.edge_type

        if edge_attr is None:
            out_subgraph.edge_attr = torch.zeros((len(edge_ids), 1), dtype=torch.float)
        else:
            out_subgraph.edge_attr = subgraph.edge_attr

        if raw_texts:
            out_subgraph.raw_texts = [raw_texts[idx] for idx in node_ids]
        else:
            out_subgraph.raw_texts = [f"{name} node {nid}" for nid in node_ids]

        if relation_texts is not None and subgraph.edge_type is not None:
            uniq_types = torch.unique(subgraph.edge_type)
            type_map = {int(t): i for i, t in enumerate(uniq_types.tolist())}
            out_subgraph.edge_type = torch.tensor([type_map[int(t)] for t in subgraph.edge_type])
            out_subgraph.relation_texts = (
                [relation_texts[t] for t in uniq_types.tolist()]
                if relation_texts is not None else ["to"] * len(uniq_types))
        else:
            out_subgraph.relation_texts = ['to']

        out_subgraph.name = name
        del subgraph
        yield out_subgraph


def select_dimension_alignment(compress_function):
    """Select dimension alignment function"""
    alignment_funcs = {
        "pca": "compress_pca",
        "svd": "compress_svd",
    }
    if compress_function in alignment_funcs:
        return getattr(compress_func, alignment_funcs[compress_function])
    else:
        raise ValueError(f"Unknown compress function: {compress_function}")


def compress_features(data_list: List[Union[Data]],  compress_fc=None, k=50) -> List[Union[Data]]:
    if compress_fc is not None:
        for data in data_list:
            if data.x.size(0) < k:
                linear = nn.Linear(data.x.shape[1], k)
                data.x = linear(data.x)
            else:
                data.x = compress_fc(data.x, k=k)
            if data.edge_attr is not None:
                data.edge_attr = compress_fc(data.edge_attr, k=k)
    return data_list


def pretrain_sampler(generators, compressed_data=None, batch_size=1024, num_workers=0) -> List[Data]:
    """Sample and process graphs for pretraining.
    Args:
        generators: Dictionary of dataset generators
        compressed_data: Pre-compressed data cache (optional)
        num_workers: The number of sub-processes for data loading
    """
    proc_graphs = []
    for name, generator in generators.items():
        dataset = generator()
        if len(dataset) == 1:
            data = dataset[0]
            if isinstance(data, Data):
                data = create_x(data)
            elif isinstance(data, HeteroData):
                data = hetero_to_data(data)
            elif isinstance(data, TemporalData):
                data = temporal_to_data(data)
            else:
                raise TypeError(f"Unsupported data format ({name}): {type(data)}")
            for sub_data in data_sampler(data, name, (5, 5, 3, 2), batch_size=batch_size, num_workers=num_workers):
                proc_graphs.append(sub_data)
        elif len(dataset) > 1:
            data = multi_to_one(dataset)
            for sub_data in multidata_sampler(data, name, batch_size=batch_size, num_workers=num_workers):
                proc_graphs.append(sub_data)
        else:
            raise ValueError(f"Dataset {name} length < 1: {len(dataset)}")
    return proc_graphs


def pretrain_loader(pretrain_dict, max_nodes=80000, batch_size=1024, num_workers=0, compress_fc=None, k=50, compressed_data=None):
    samplers = pretrain_sampler(pretrain_dict, compressed_data=None, batch_size=batch_size, num_workers=num_workers)
    return compress_features(samplers, compress_fc=compress_fc, k=k)
    #return general_loader(samplers, max_nodes, num_workers=num_workers, compress_fc=compress_fc, k=k)

def pyg_to_dgl(data: Data) -> dgl.DGLGraph:
    """Convert a PyG Data subgraph to a DGLGraph subgraph"""
    src, dst = data.edge_index[0], data.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=data.num_nodes)
    g.ndata["feat"] = data.x
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        g.edata["feat"] = data.edge_attr
    if hasattr(data, "edge_type") and data.edge_type is not None:
        g.edata["edge_type"] = data.edge_type

    return g


def compute_spd_matrix(graph, k: int = 5, device: str = "cpu"):
    """
    Graphormer-style k-hop shortest path distance (undirected).
    Distances > k are clipped to k+1.
    """
    num_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src = src.tolist()
    dst = dst.tolist()

    # build undirected adjacency
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    # initialize SPD with k+1
    spd = torch.full((num_nodes, num_nodes), k + 1, dtype=torch.float32)
    spd.fill_diagonal_(0)

    for start in range(num_nodes):
        q = deque([(start, 0)])
        visited = {start}

        while q:
            u, d = q.popleft()
            if d == k:
                continue

            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    spd[start, v] = d + 1
                    q.append((v, d + 1))

    return spd.to(device)

def run(params):
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    input_dim = params["input_dim"]

    # Initialize model
    model = UniGraph2(
        input_dims={"text": input_dim},
        hidden_dim=input_dim,
        num_experts=8,
        num_selected_experts=2,
        num_layers=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    compress_func = select_dimension_alignment(params["compress_function"])

    for epoch in range(params["epochs"]):
        graph_list = pretrain_loader(PRETRAIN_DICT[params['exp']], max_nodes=1500, batch_size=params["batch_size"], num_workers=2, compress_fc=compress_func, k=input_dim, compressed_data=None)

        model.train()
        total_train_loss = 0

        for graph_data in graph_list:
            graph = pyg_to_dgl(graph_data).to(device)
            graph = dgl.add_self_loop(graph)
            features = {"text": graph.ndata["feat"]}

            spd_matrix = compute_spd_matrix(graph, k=2, device=device).to(device)
            loss, _ = model(graph, features, spd_matrix)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            print(f'loss={loss.item():.4f}')
            total_train_loss += loss.item()

        scheduler.step()
        
        torch.save(model.state_dict(), "checkpoints/unigraph2_pretrain_exp4.pt")
        print(f"[Epoch {epoch}] Train Loss: {total_train_loss:.4f}")


if __name__ == "__main__":
    params = get_args()
    run(params)