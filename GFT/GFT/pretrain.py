import os
import sys
import os.path as osp
import yaml
from copy import deepcopy
from typing import List, Dict, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader
from utils.maxnode_loader import MultiGraphLoader
from torch_geometric.data import HeteroData, TemporalData, Data, Batch
from model.encoder import Encoder, InnerProductDecoder
from model.pt_model import PretrainModel
from model.vq import VectorQuantize
from utils.args import get_args_pretrain
from utils.others import seed_everything, get_scheduler, get_device_from_model, check_path, create_x
from utils.compress_func import compress_pca
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils.tools import EarlyStopping
from utils import compress_func
from utils.subbatch_loader import BatchGraphLoader
from torch_sparse import SparseTensor
from itertools import chain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from data_provider.data_generator import *

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
        subgraph.raw_texts = ['N/A'] * subgraph.num_nodes
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

    #x = data.x if not (data.x.layout == torch.sparse_csr or data.x.layout == torch.sparse_coo
    #                   ) else data.x.to_dense()  # Ensure x is dense for NeighborLoader
    raw_texts = data.raw_texts if hasattr(data, 'raw_texts') else None
    edge_type = data.edge_type if hasattr(data, 'edge_type') else None
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    relation_texts = data.relation_texts if hasattr(data, 'relation_texts') else None
    # relation_texts = [data.relation_texts[type] for type in edge_type] if hasattr(data, 'relation_texts') else None

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
            out_subgraph.raw_texts = ['N/A'] * len(node_ids)  # ['' for _ in range(len(node_ids))]

        if relation_texts is not None and subgraph.edge_type is not None:
            # Here the relation_texts is mapping to each edge from edge_type
            out_subgraph.relation_texts = relation_texts
        else:
            out_subgraph.relation_texts = ['to'] * subgraph.num_edges

        out_subgraph.name = name
        del subgraph
        yield out_subgraph


def select_dimension_alignment(params):
    """Select dimension alignment function"""
    alignment_funcs = {
        "pca": "compress_pca",
        "svd": "compress_svd",
    }
    if params['compress_function'] in alignment_funcs:
        return getattr(compress_func, alignment_funcs[params['compress_function']])
    else:
        raise ValueError(f"Unknown compress function: {params['compress_function']}")


def pretrain_sampler(generators, compressed_data=None, batch_size=1024, num_workers=0) -> List[Data]:
    """
    Sample and process graphs for pretraining.

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
            for sub_data in data_sampler(data, name, (10, 10, 10, 10), batch_size=batch_size, num_workers=num_workers):
                proc_graphs.append(sub_data)
        elif len(dataset) > 1:
            data = multi_to_one(dataset)
            for sub_data in multidata_sampler(data, name, batch_size=batch_size, num_workers=num_workers):
                proc_graphs.append(sub_data)
        else:
            raise ValueError(f"Dataset {name} length < 1: {len(dataset)}")
    return proc_graphs


def general_loader(sampler: List[Union[Data]], max_nodes, num_workers=0, compress_fc=None, k=50) -> MultiGraphLoader:
    def _pad_edge_attr(data_list: List[Union[Data]]) -> List[Union[Data]]:
        # edge attr alignment by padding
        if not data_list:
            return []
        max_edge_dim = max(
            (
                data.edge_attr.size(1)
                for data in data_list
                if hasattr(data, "edge_attr") and data.edge_attr is not None
            ),
            default=1,
        )
        for data in data_list:
            # alignment edge feature
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                pad_edge = max_edge_dim - data.edge_attr.size(1)
                if pad_edge > 0:
                    data.edge_attr = F.pad(data.edge_attr, (0, pad_edge), mode="constant", value=0)
            else:
                num_edges = data.edge_index.size(1)
                data.edge_attr = torch.zeros((num_edges, max_edge_dim), dtype=torch.float)
        return data_list
    
    def _re_idx_edge_type(data_list: List[Union[Data]]) -> List[Union[Data]]:
        # Re-index each data.item’s edge_type by adding the current offset, 
        # so that edge_type values across the list are globally unique and align with the concatenated relation_texts.
        curr_type_offset = 0
        for data in data_list:
            data.edge_type = data.edge_type + curr_type_offset
            curr_type_offset += len(data.relation_texts)
        return data_list

    def _compress_features(data_list: List[Union[Data]]) -> List[Union[Data]]:
        if compress_fc is not None:
            for data in data_list:
                data.x = compress_fc(data.x, k=k)
                if data.edge_attr is not None:
                    data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
                    data.edge_attr = compress_fc(data.edge_attr, k=k)
        return data_list

    loader = MultiGraphLoader(sampler, max_nodes, shuffle=True, num_workers=num_workers)
    for data_list in loader:
        data_list = _pad_edge_attr(data_list)
        data_list = _re_idx_edge_type(data_list)
        data_list = _compress_features(data_list)
        data = Batch.from_data_list(data_list)
        if hasattr(data, "raw_texts") and data.raw_texts is not None:
            # search raw_texts of node 'n[i]' by data.raw_texts[i]
            data.raw_texts = list(chain.from_iterable(data.raw_texts))
        if hasattr(data, "relation_texts") and data.relation_texts is not None:
            # search relation_texts of edge 'e[i]' by data.relation_texts[data.edge_type[i]]
            data.relation_texts = list(chain.from_iterable(data.relation_texts))
        if hasattr(data, "name") and data.name is not None:
            data.name = [data.name[i] for i in data.batch]
        yield data


def pretrain_loader(pretrain_dict, max_nodes=80000, batch_size=1024, num_workers=0, compress_fc=None, k=50, compressed_data=None):
    samplers = pretrain_sampler(pretrain_dict, compressed_data=None, batch_size=batch_size, num_workers=num_workers)
    return general_loader(samplers, max_nodes, num_workers=num_workers, compress_fc=compress_fc, k=k)


def pretrain(model, loader, optimizer, params, scheduler=None, no_codebook=False):
    model.train()
    device = get_device_from_model(model)

    #for step, data in enumerate(loader):
    step = 0
    for batch_idx, data in enumerate(loader, start=1):
        bs = data.batch_size
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        graph = [x, edge_index, edge_attr]

        aug_x, _ = mask_feature(x, p=params['feat_p'])
        aug_edge_index, aug_edge_attr = dropout_adj(
            edge_index, edge_attr, p=params['edge_p'],
            force_undirected=True, num_nodes=x.size(0)
        )
        aug_graph = [aug_x, aug_edge_index, aug_edge_attr]

        z, quantize, indices, losses = model(
            aug_graph, graph, params['topo_recon_ratio'], bs=bs, no_codebook=no_codebook
        )

        feat_recon_loss = params['feat_lambda'] * losses['feat_recon_loss']
        topo_recon_loss = params['topo_lambda'] * losses['topo_recon_loss']
        topo_sem_recon_loss = params['topo_sem_lambda'] * losses['topo_sem_recon_loss']
        sem_recon_loss = params['sem_lambda'] * losses['sem_recon_loss']
        commit_loss = losses['commit_loss']
        loss = feat_recon_loss + topo_recon_loss + topo_sem_recon_loss + sem_recon_loss + commit_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        model.ema_update_sem_encoder(decay=params['sem_encoder_decay'])
        
        print(f'loss={loss.item():.4f} | '
            f'feat={feat_recon_loss.item():.4f} | '
            f'topo={topo_recon_loss.item():.4f} | '
            f'topo_sem={topo_sem_recon_loss.item():.4f} | '
            f'sem={sem_recon_loss.item():.4f} | '
            f'commit={commit_loss.item():.4f}')


def run(params):
    seed_everything(params["seed"])
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )
    vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=False,
        ema_update=False,
    )
    feat_recon_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_recon_decoder = InnerProductDecoder(hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"])
    topo_sem_recon_decoder = nn.Linear(params["hidden_dim"] * 2, params["hidden_dim"])

    pretrain_model = PretrainModel(
        encoder=encoder, vq=vq,
        feat_recon_decoder=feat_recon_decoder,
        topo_recon_decoder=topo_recon_decoder,
        topo_sem_recon_decoder=topo_sem_recon_decoder,
    ).to(device)

    optimizer = AdamW(pretrain_model.parameters(), lr=params["pretrain_lr"],
                      weight_decay=params["pretrain_weight_decay"])
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["pretrain_epochs"])
    compress_func = select_dimension_alignment(params)

    for epoch in range(params["pretrain_epochs"]):
        train_loader = pretrain_loader(PRETRAIN_DICT[params['exp']], max_nodes=60000, batch_size=params['pretrain_batch_size'], num_workers=2, compress_fc=compress_func, k=params['input_dim'], compressed_data=None)
        pretrain(model=pretrain_model, loader=train_loader,
                             optimizer=optimizer, params=params, scheduler=scheduler)
        
        save_path = osp.join(
            params['model_path'],
            f"input_dim{params['input_dim']}_hd{params['hidden_dim']}_codebook_size_{params['codebook_size']}_layer_{params['num_layers']}_"
            f"pretrain_on_exp{params['exp']}_seed_{params['seed']}"
        )
        check_path(save_path)
        try:
            pretrain_model.save_encoder(osp.join(save_path, f"encoder_{epoch}.pt"))
            pretrain_model.save_vq(osp.join(save_path, f"vq_{epoch}.pt"))
            print(f"Model saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save model at epoch {epoch}: {e}")

    print("Training finished.")


if __name__ == "__main__":
    params = get_args_pretrain()
    params['model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    print("==========  Pre-training Config  ==========")
    for k, v in sorted(params.items()):
        print(f"{k:25s}: {v}")
    print("==========================================")

    run(params)
