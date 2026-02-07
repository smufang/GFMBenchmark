import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import HeteroData, TemporalData, Data, Batch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Union
from torch_sparse import SparseTensor
import numpy as np
import scipy.sparse as sp
import random
from preprompt import PrePrompt
from utils import compress_func
from utils.subbatch_loader import BatchGraphLoader
from utils.compress_func import compress_pca
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils.maxnode_loader import MultiGraphLoader
from utils.others import create_x, complete_data
from itertools import chain
import aug
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

def get_args_pretrain():
    parser = argparse.ArgumentParser("MDGPT")
    parser.add_argument('--dataset', type=str, default="Cora", help='data')
    parser.add_argument('--seed', type=int, default=39, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--input_dim', type=int, default=50)
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--premodel', type=str, default='DGI', help='the type of pretrain model')   
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--compress_function', type=str, default='pca',
                            help='dimension alignment method: pca/svd')
    args = parser.parse_args()

    print(args)
    print('-' * 10)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return vars(args)


dataset_mapping = {
    'Cora': cora,
    'ACM': acm,
    'Reddit': reddit,
    'Wisconsin': wisconsin,
    'Elliptic': elliptic,
    'Photo': photo,
    'HIV': hiv,
    'COX2': cox2,
    'PROTEINS': proteins,
    'FB15K-237': fb15k237,
}

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


def get_data(params):
    data_name = params['dataset']
    if data_name in dataset_mapping:
        dataset = dataset_mapping[data_name]()
    else:
        raise ValueError(f"unknown data: {data_name}")

    compress_func = select_dimension_alignment(params)
    if len(dataset) > 1:
        data = multi_to_one(dataset)
        params['data_type'] = "multi"
    else:
        data = dataset[0]
        if isinstance(data, TemporalData):
            data = temporal_to_data(data)
            params['data_type'] = "temporal"
        elif isinstance(data, HeteroData):
            data = hetero_to_data(data, need_y=True, use_embedding=False)
            params['data_type'] = "hetero"
        elif isinstance(data, Data):
            data = create_x(data)
            params['data_type'] = "graph"
        else:
            raise ValueError(f"Unknown data type: {type(data)}")
    data = complete_data(data, data_name)
    data.x = compress_func(data.x, k=params['input_dim'])
    if data.edge_attr is not None:
        data.edge_attr = compress_func(data.edge_attr, k=params["input_dim"])
    return data


def run(params):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    save_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f'id{params["input_dim"]}_hd{params["hid_units"]}_{params["premodel"]}_{params["dataset"]}.pt')
    
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    model = PrePrompt(params["input_dim"], params["hid_units"], num_layers_num=3, dropout=0.1, premodel=params["premodel"]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0.0)

    dataset = get_data(params).to(device)
    best_loss = 1e9
    cnt_wait = 0
    for epoch in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = model(dataset.x, dataset.edge_index)
        total_loss.backward()
        optimizer.step()

        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == params["patience"]:
            print('Early stopping!')
            break
        
        print(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    params = get_args_pretrain()
    run(params)