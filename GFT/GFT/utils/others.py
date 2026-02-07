import datetime
from lib2to3.pytree import BasePattern
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path
from torch_geometric.data import Data
import torch
from torch_geometric.utils import (degree, remove_self_loops, add_self_loops, to_undirected, k_hop_subgraph, coalesce,
                                   to_edge_index, to_torch_coo_tensor, is_undirected, to_dense_adj)

from model.encoder import Encoder
from model.vq import VectorQuantize

EPS = 1e-6


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def get_mask(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def flip_edges(data, p=0.2):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip_edges = int(num_edges * p / 2)
    else:
        num_flip_edges = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]

    flipped_edges = torch.randint(0, num_nodes, size=(num_flip_edges, 2))

    for n1, n2 in flipped_edges:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    edge_index = adj.to_sparse().coalesce().indices()
    data.edge_index = edge_index
    data.edge_attr = None
    return data


def get_device(params, optimized_params=None):
    if optimized_params is None or len(optimized_params) == 0:
        device = torch.device(f"cuda:{params['device']}")
    else:
        device = torch.device(f"cuda")
    return device


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def get_device_from_model(model):
    return next(model.parameters()).device


def active_code(encoder, vq, data):
    z = encoder(data.x, data.edge_index, data.edge_attr)
    _, indices, _, _ = vq(z)
    codebook_size = vq.codebook_size
    codebook_head = vq.heads
    return indices.unique(), indices.unique().numel() / (codebook_size * codebook_head)


def load_params(model, path):
    if isinstance(model, Encoder):
        model.load_state_dict(torch.load(path))
    elif isinstance(model, VectorQuantize):
        z = torch.randn(100, model.dim)
        model(z)
        model.load_state_dict(torch.load(path))
    return model


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize(embedding, label=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label, cmap='tab10')
    plt.show()


def sample_proto_instances(labels, split, num_instances_per_class=10):
    y = labels.cpu().numpy()
    target_y = y[split]
    classes = np.unique(target_y)

    class_index = []
    for i in classes:
        c_i = np.where(y == i)[0]
        c_i = np.intersect1d(c_i, split)
        class_index.append(c_i)

    proto_idx = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        proto_idx = np.concatenate((proto_idx, idx[:num_instances_per_class]))

    return proto_idx.astype(int)


def sample_proto_instances_for_graph(labels, split, num_instances_per_class=10):
    y = labels
    ndim = y.ndim
    if ndim == 1:
        y = y.reshape(-1, 1)

    # Map class and instance indices

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    target_y = y[split]
    task_list = target_y.shape[1]

    # class_index_pos = {}
    # class_index_neg = {}
    task_index_pos, task_index_neg = [], []
    for i in range(task_list):
        c_i = np.where(y[:, i] == 1)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_pos.append(c_i)

        c_i = np.where(y[:, i] == 0)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_neg.append(c_i)

    assert len(task_index_pos) == len(task_index_neg)

    # Randomly select instances for each task

    proto_idx, proto_labels = {}, {}
    for task, (idx_pos, idx_neg) in enumerate(zip(task_index_pos, task_index_neg)):
        tmp_proto_idx, tmp_labels = np.array([]), np.array([])

        # Randomly select instance for the task

        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        idx_pos = idx_pos[:num_instances_per_class]
        idx_neg = idx_neg[:num_instances_per_class]

        # Store the randomly selected instances

        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_pos))
        tmp_labels = np.concatenate((tmp_labels, np.ones(len(idx_pos))))
        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_neg))
        tmp_labels = np.concatenate((tmp_labels, np.zeros(len(idx_neg))))

        proto_idx[task] = tmp_proto_idx.astype(int)
        proto_labels[task] = tmp_labels.astype(int)

    return proto_idx, proto_labels


def mask2idx(mask):
    return torch.where(mask == True)[0]

def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def create_x(data: Data) -> Data:
    '''create node features by one hot if not exist'''
    if hasattr(data, "x") and data.x is not None:
        return data
    elif hasattr(data, "num_nodes") and data.num_nodes is not None:
        data.x = torch.eye(data.num_nodes) if data.num_nodes <= 1000 else torch.eye(
            data.num_nodes).to_sparse()  # torch.randn(data.num_nodes, f_size)
        return data
    else:
        raise AttributeError("No x or num_nodes")


def complete_data(data: Data, name, need_y=False) -> Data:
    '''complete data attributes'''
    new_data = Data()
    new_data.x = data.x
    new_data.edge_index = data.edge_index
    new_data.name = name
    if not hasattr(data, 'edge_type') or data.edge_type is None:
        new_data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
    else:
        new_data.edge_type = data.edge_type

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        new_data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float)
    else:
        new_data.edge_attr = data.edge_attr

    if not hasattr(data, 'raw_texts') or data.raw_texts is None:
        new_data.raw_texts = ['N/A'] * data.num_nodes
    else:
        new_data.raw_texts = data.raw_texts

    if not hasattr(data, 'relation_texts') or data.relation_texts is None:
        new_data.relation_texts = ['to']
    else:
        new_data.relation_texts = data.relation_texts

    if hasattr(data, 'batch') and data.batch is not None:
        new_data.batch = data.batch

    if need_y and hasattr(data, 'y') and data.y is not None:
        if not hasattr(data, 'y') or data.y is None:
            raise ValueError(f"Data {name} has no labels 'y' but need_y is True.")
        else:
            new_data.y = data.y
            
            if hasattr(data, 'label_names') and data.label_names is not None:
                new_data.label_names = data.label_names

            if hasattr(data, 'label_descs') and data.label_descs is not None:
                new_data.label_descs = data.label_descs

    return new_data