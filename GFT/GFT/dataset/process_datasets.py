import argparse
import os
import os.path as osp

os.sys.path.append(os.path.dirname(__file__))

import torch
from pytorch_lightning.loggers import WandbLogger
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit
from gp.lightning.module_template import ExpConfig
from types import SimpleNamespace
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge

from torchmetrics import AUROC, Accuracy
from data_utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
    ENCODER_DIM_DICT,
)

from task_constructor import UnifiedTaskConstructor

print(os.path.dirname(__file__))

WEIGHT = load_yaml(os.path.join(os.path.dirname(__file__), '..', '..', "config", "pt_data.yaml"))
datasets = {k: v.keys() for k, v in WEIGHT.items()}

params = load_yaml(os.path.join(os.path.dirname(__file__), "processed_params.yaml"))
params = SimpleNamespace(**params)

task_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "task_config.yaml"))
data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))


# These functions are for pre-training


def refine_dataset(dataset):
    # works for molecule graphs
    if dataset.data.get("node_embs") is not None:
        dataset.data.node_text_feat = dataset.data.node_embs
        dataset.data.node_embs = None
    if dataset.data.get("edge_embs") is not None:
        dataset.data.edge_text_feat = dataset.data.edge_embs
        dataset.data.edge_embs = None
    if dataset.data.get("pretrain_edge_index") is not None:
        dataset.data.edge_index = dataset.data.pretrain_edge_index
        dataset.data.pretrain_edge_index = None
    return dataset


def filter_unnecessary_attrs(dataset, mode="pretrain"):
    keys = [
        "x",
        "xe",
        "edge_index",
        "node_text_feat",
        "edge_text_feat",
        "class_node_text_feat",
    ]

    if mode == 'pretrain':
        keys = [
            "x",
            "xe",
            "edge_index",
            "node_text_feat",
            "edge_text_feat",
        ]

    for k, v in dataset.data.to_dict().items():
        if k not in keys:
            dataset.data[k] = None
    return dataset


def span_node_and_edge_idx(dataset):
    # Define node index
    if dataset.data.x.ndim == 1:
        return dataset

    num_nodes = dataset.data.x.shape[0]
    dataset.data.x = torch.arange(num_nodes)

    # Define edge index
    num_edge_types = dataset.data.edge_text_feat.shape[0]
    num_edges = dataset.data.edge_index.shape[1]

    if num_edge_types == 1:
        dataset.data.xe = torch.zeros([num_edges], dtype=torch.long)
    else:
        dataset.data.xe = dataset.data.edge_types
    return dataset


def get_task_constructor(data_path):
    # Load processed_params.yaml
    encoder = SentenceEncoder(params.llm_name, batch_size=params.llm_b_size)

    if isinstance(params.task_names, str):
        task_names = [a.strip() for a in params.task_names.split(",")]
    else:
        task_names = params.task_names

    root = data_path
    if params.llm_name != "ST":
        root = f"{data_path}_{params.llm_name}"

    tasks = UnifiedTaskConstructor(
        task_names,
        encoder,
        task_config_lookup,
        data_config_lookup,
        root=root,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
    )

    return tasks


def idx2mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[idx] = True
    return mask


def mask2idx(mask):
    return torch.where(mask)[0]


def get_pt_data(data_path, setting="all"):
    from torch_geometric.data import Batch

    if isinstance(setting, list):
        dataset_names = []
        for s in setting:
            dataset_names.extend(datasets.get(s, s))
    elif isinstance(setting, str):
        dataset_names = datasets.get(setting, setting)

    print(f"Pre-training on {dataset_names}")

    tasks = get_task_constructor(data_path)
    dataset_list = []

    for dataset_name in dataset_names:
        data_config = data_config_lookup[dataset_name]
        dataset = tasks.get_ofa_data(data_config)
        dataset = refine_dataset(dataset)
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
        dataset_list.append(dataset.data)

    def preprocess_dataset_list(dataset_list):
        x_start, xe_start = 0, 0
        for dataset in dataset_list:
            num_unique_nodes = dataset.node_text_feat.shape[0]
            num_edge_types = dataset.edge_text_feat.shape[0]
            dataset.x += x_start
            dataset.xe += xe_start
            x_start += num_unique_nodes
            xe_start += num_edge_types
        return dataset_list

    dataset_list = preprocess_dataset_list(dataset_list)
    pretrain_dataset = Batch.from_data_list(dataset_list)
    return pretrain_dataset


def get_train_node_idx(data, weights):
    assert data.ptr is not None

    total_idx = torch.tensor([], dtype=torch.long)
    for idx, (s, e) in enumerate(zip(data.ptr[:-1], data.ptr[1:])):
        arr = torch.arange(s, e)
        int_weight, mod_weight = int(weights[idx]), weights[idx] - int(weights[idx])

        left_idx = arr.repeat(int_weight)
        right_idx = arr[torch.randperm(arr.size(0))[: int(mod_weight * arr.size(0))]]
        idx = torch.cat([left_idx, right_idx])
        total_idx = torch.cat([total_idx, idx])
    return total_idx

def preprocess_split(split):
    if isinstance(split, dict):
        split_list = []
        if isinstance(split["test"], list):
            for train, valid, test in zip(split["train"], split["valid"], split["test"]):
                split_list.append({"train": train, "valid": valid, "test": test})
        elif split["test"].ndim == 1:
            for train, valid in zip(split["train"], split["valid"]):
                split_list.append({"train": train, "valid": valid, "test": split["test"]})

        return split_list

def get_node_data(tasks, dataset_name):
    data_config = data_config_lookup[dataset_name]
    dataset = tasks.get_ofa_data(data_config)
    data = dataset[0]
    num_tasks = 1
    if dataset_name in ["cora", "pubmed"]:
        split = {"train": data.train_masks, "valid": data.val_masks, "test": data.test_masks}
        split = preprocess_split(split)
        labels = data.y
        num_classes = labels.unique().shape[0]

    elif dataset_name in ["wikics"]:
        split = {"train": data.train_mask.T, "valid": data.val_mask.T, "test": data.test_mask.T}
        split = preprocess_split(split)
        labels = data.y
        num_classes = labels.unique().shape[0]

    elif dataset_name in ["arxiv"]:
        split = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}
        labels = data.y.squeeze()
        num_classes = labels.unique().shape[0]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for node classification task")
    return dataset, split, labels, num_classes, num_tasks

def get_link_data(tasks, dataset_name):
    if dataset_name in ["WN18RR", "FB15K237"]:
        data_config = data_config_lookup[dataset_name]
        dataset = tasks.get_ofa_data(data_config)
        split = tasks.get_data_split(data_config)
        num_tasks = 1
        data = dataset[0]
        labels = data.edge_types
        num_classes = labels.unique().shape[0]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for link classification task")
    return dataset, split, labels, num_classes, num_tasks

def get_graph_clf_graph(tasks, dataset_name):
    data_config = data_config_lookup[dataset_name]
    dataset = tasks.get_ofa_data(data_config)
    split = tasks.get_data_split(data_config)
    if dataset_name in ["chemhiv"]:
        num_tasks = 1
        num_classes = None
        labels = dataset.y
    elif dataset_name in ["chempcba"]:
        num_tasks = 128
        num_classes = None
        labels = dataset.y.reshape(-1, num_tasks)
    elif dataset_name in ["chemblpre"]:
        raise NotImplementedError(f"Dataset {dataset_name} is only used for pre-training")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for graph classification task")

    return dataset, split, labels, num_classes, num_tasks

# Statistics
# Cora: train_masks (idx): 140, val_masks (idx): 500, test_masks (idx): Remaining ---- 10 splits
# Pubmed: train_masks (idx): 60, val_masks (idx): 500, test_masks (idx): Remaining ---- 10 splits
# WikiCS: train_mask (mask): 5%, val_mask (mask): 15%, test_mask (mask, fix): 50% ---- 20 splits
# Arxiv: train_mask (mask): public, val_mask (mask): public, test_mask (mask, fix): public ---- 1 split

# WN18RR: split['train']: 86835, split['valid']: 3034, split['test']: 3134 ---- 1 split
# FB15K237: split['train']: 272115, split['valid']: 17535, split['test']: 20466 ---- 1 split

# chempcba: split['train']: public, split['valid']: public, split['test']: public ---- 1 split
# chemhiv: split['train']: public, split['valid']: public, split['test']: public ---- 1 split
# chemblpre: only used for pre-training

def get_finetune_graph(data_path, dataset_name):
    tasks = get_task_constructor(data_path)

    if dataset_name in ["cora", "pubmed", "wikics", "arxiv"]:
        return get_node_data(tasks, dataset_name)
    elif dataset_name in ["WN18RR", "FB15K237"]:
        return get_link_data(tasks, dataset_name)
    elif dataset_name in ["chemhiv", "chemblpre", "chempcba"]:
        return get_graph_clf_graph(tasks, dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")


if __name__ == "__main__":
    pretrain_graph = get_pt_data(["arxiv", "node"])
    print(pretrain_graph)
