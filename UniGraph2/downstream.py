import os
import sys
import argparse
import random
import dgl
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
from torch_geometric.data import Data, HeteroData, TemporalData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean
from torchmetrics import Accuracy
from utils.subbatch_loader import BatchGraphLoader
from sklearn.metrics import f1_score
from models.unigraph2 import UniGraph2
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils import compress_func
from utils.utils import create_x, complete_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

def get_args_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_dataset", "--dataset", "--data", type=str, default="Pubmed")
    parser.add_argument("--exp", type=int, default=3)
    parser.add_argument('--compress_function', type=str, default='pca',
                        help='dimension alignment method: pca/svd')
    parser.add_argument('--input_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--task_name", type=str, default="node") #edge, graph
    parser.add_argument("--n_task", type=int, default=50)
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return vars(args)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

dataset_mapping = {
    ### exp1 node
    'Pubmed': pubmed,
    'ogbn-mag': ogbn_mag,
    'Wikipedia': wikipedia,
    'Actor': actor,
    'Chameleon': chameleon,
    'Products': products,
    'ogbn-proteins': ogbn_proteins,
    'T-Finance': tfinance,
    'DGraph': dgraph,
    #### exp1 graph
    'PCBA': pcba,
    'BZR': bzr,
    ### exp1 edge
    'WIKI': wiki,
    'WN18RR': wn18rr,
    ### exp2 node
    'Cora': cora,
    'ACM': acm,
    'Reddit': reddit,
    'Wisconsin': wisconsin,
    'Elliptic': elliptic,
    'Photo': photo,
    ### exp2 graph
    'HIV': hiv,
    'COX2': cox2,
    'PROTEINS': proteins,
    ### exp2 edge
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
    data_name = params['finetune_dataset']
    if data_name in dataset_mapping:
        dataset = dataset_mapping[data_name]()
    else:
        raise ValueError(f"unknown dataset: {data_name}")

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
        data.edge_attr = compress_func(data.edge_attr, k=params["hidden_dim"])
    return data

def get_loader(params, data, task_labels, task_ids, is_train):
    batch_size = params['batch_size']
    persistent_workers=False
    if params['task_name'] == "node":
        loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5],
            input_nodes=task_ids,
            batch_size=batch_size,
            shuffle=False,
            directed=False,
            replace=False,
            num_workers=params['num_workers'],
            persistent_workers=persistent_workers
        )
        for batch in loader:
            batch_labels = task_labels[batch.input_id]
            batch_ids = torch.arange(len(batch_labels), dtype=torch.long)
            yield batch, batch_labels, batch_ids

    elif params['task_name'] == "edge":
        loader = LinkNeighborLoader(
            data,
            num_neighbors=[5, 5, 5],
            edge_label_index=data.edge_index[:, task_ids],
            edge_label=task_labels,
            batch_size=batch_size,
            shuffle=False,
            directed=False,
            replace=False,
            num_workers=params['num_workers'],
            persistent_workers=persistent_workers
        )
        for batch in loader:
            yield batch, batch.edge_label, batch.edge_label_index

    elif params['task_name'] == "graph":
        loader = BatchGraphLoader(
            data,
            graph_label_index=task_ids,
            graph_label=task_labels,
            batch_size=batch_size,
            shuffle=False,
            num_workers=params['num_workers']
        )
        for batch in loader:
            yield batch, batch.graph_label, batch.graph_label_index


def get_unigraph2_encoder(ckpt_path: str, in_dim_text: int, hidden_dim: int, device: torch.device):
    input_dims = {
        "text": in_dim_text,
    }

    model = UniGraph2(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_experts=8,
        num_selected_experts=2,
        num_layers=3,
        feat_drop_rate=0.1,
        edge_mask_rate=0.1,
        gamma=2.0,
        lambda_spd=0.5,
    )

    state = torch.load(ckpt_path)
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    else:
        state_dict = state

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def encoder(graph, features):
        return model.encoder(graph, features)
    return encoder


def compute_node_embeddings(encoder, graph, text_feats, image_feats=None):
    with torch.no_grad():
        feats = {"text": text_feats}
        if image_feats is not None:
            feats["image"] = image_feats
        h = encoder(graph, feats) 
    return h


def compute_prototypes(emb: torch.Tensor, labels: torch.Tensor, device):
    proto_classes = torch.unique(labels)
    prototypes = []
    for c in proto_classes.tolist():
        mask = (labels == c)
        prototypes.append(emb[mask].mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    return prototypes, proto_classes.to(device)


def prototype_predict(emb: torch.Tensor, prototypes: torch.Tensor, proto_classes: torch.Tensor):
    sim = F.cosine_similarity(emb.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
    idx = sim.argmax(dim=-1)
    pred = proto_classes[idx]
    return pred


def run_model_graph(model, full_data, task, params, is_train, proto_emb, proto_classes, is_batch):
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    data = Data()
    data.x = full_data.x.contiguous()
    data.edge_index = full_data.edge_index.contiguous()
    data.edge_attr = full_data.edge_attr.contiguous()
    if params['data_type'] == "temporal":
        from torch_geometric.utils import to_undirected
        data.edge_index, data.edge_attr = to_undirected(data.edge_index, edge_attr=data.edge_attr, reduce='mean')
    if hasattr(full_data, "batch") and params['task_name'] == "graph":
        data.batch = full_data.batch  # Graph-level

    task_labels = torch.tensor(task["labels"])
    task_ids = torch.tensor(task["idx"])
    if params['task_name'] == 'edge':
        ids = torch.arange(task_labels.size(0), device=task_labels.device)

    if not is_batch:
        if params['task_name'] == 'edge':
            task_ids = data.edge_index[:, task_ids]
        loader = [(data, task_labels, task_ids)]
    else:
        loader = get_loader(params, data, task_labels, task_ids, is_train)

    num_classes = params["num_classes"]

    if is_train:
        proto_emb = []
        emb_buffer, label_buffer = [], []
        for batch, batch_labels, batch_ids in loader:
            try:
                graph = dgl.graph((batch.edge_index[0], batch.edge_index[1]), num_nodes=batch.x.size(0)).to(device)
                graph.ndata['feat'] = batch.x.to(device)
                graph = dgl.add_self_loop(graph)

                emb = compute_node_embeddings(model, graph, text_feats=graph.ndata["feat"], image_feats=None)
                if params['task_name'] == 'graph':
                    emb = scatter_mean(emb, batch.batch.to(device), dim=0)
                
                emb = emb[batch_ids]
                if params['task_name'] == 'edge':
                    emb = emb.mean(dim=0)
                
                emb_buffer.append(emb.detach())
                label_buffer.append(batch_labels)

            except Exception as e:
                if not is_batch:
                    print(f"{e}\n Switching to batch mode.")
                    is_batch = True
                    return run_model_graph(model, full_data, task, params, is_train, proto_emb, proto_classes, is_batch)
                else:
                    print(f"Num Nodes: {batch.num_nodes}")
                    print(f"Num Edges: {batch.num_edges}")
                    raise e
            
            del batch, batch_labels, batch_ids, graph, emb
            torch.cuda.empty_cache()
        
        task_emb = torch.cat(emb_buffer, dim=0)
        labels = torch.cat(label_buffer, dim=0)
        prototypes, proto_classes = compute_prototypes(task_emb, labels, device)
        return prototypes, proto_classes

    else: #test
        pred_list, y_list = [], []
        with torch.no_grad():
            for batch, batch_labels, batch_ids in loader:
                try:
                    graph = dgl.graph((batch.edge_index[0], batch.edge_index[1]), num_nodes=batch.x.size(0)).to(device)
                    graph.ndata['feat'] = batch.x.to(device)
                    graph = dgl.add_self_loop(graph)

                    emb = compute_node_embeddings(model, graph, text_feats=graph.ndata["feat"], image_feats=None)
                    
                    if params['task_name'] == 'graph':
                        emb = scatter_mean(emb, batch.batch.to(device), dim=0)

                    emb = emb[batch_ids]
                    if params['task_name'] == 'edge':
                        emb = emb.mean(dim=0)

                    y = batch_labels.to(device)
                    pred = prototype_predict(emb, proto_emb, proto_classes)

                    pred_list.append(pred)
                    y_list.append(y)

                except Exception as e:
                    if not is_batch:
                        print(f"{e}\n Switching to batch mode.")
                        is_batch = True
                        return run_model_graph(model, full_data, task, params, is_train, proto_emb, proto_classes, is_batch)
                    else:
                        print(f"Num Nodes: {batch.num_nodes}")
                        print(f"Num Edges: {batch.num_edges}")
                        raise e
                
                del batch, batch_labels, batch_ids, graph, emb, pred
                torch.cuda.empty_cache()
        
        preds = torch.cat(pred_list, dim=0)
        labels = torch.cat(y_list, dim=0)
        return preds, labels


def main(params):
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(params['seed'])
    print(f"device: {device}")
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", f"unigraph2_pretrain_exp{params['exp']}.pt")

    is_batch = True

    fewshot_base_dir = os.path.join(BASE_DIR, "..", "datasets_split", params['finetune_dataset'], "few-shot")
    fewshot_tasks_dict = {}
    for folder in os.listdir(fewshot_base_dir):
        if folder.startswith(f"{params['task_name']}-"):
            task_dim = int(folder.split('-')[-1])
            folder_path = os.path.join(fewshot_base_dir, folder)
            tasks = []
            for file in os.listdir(folder_path):
                if file.endswith(".pt") and f"_{params['n_shot']}-shot_{params['n_task']}-tasks" in file:
                    tasks += torch.load(os.path.join(folder_path, file))
            fewshot_tasks_dict[task_dim] = tasks
        else:
            raise ValueError(f"Wrong task name: {params['task_name']}")

    split_dir = os.path.join(".", "datasets_split", params['finetune_dataset'], "split")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"No split directory found: {split_dir}")
    test_split_dict = {}
    for file in os.listdir(split_dir):
        if file.startswith(f"split_{params['task_name']}_") and file.endswith(".pt"):
            task_dim = int(file.split('_')[-1].split('.')[0])
            split_dict = torch.load(os.path.join(split_dir, file))
            test_split_dict[task_dim] = {
                'idx': split_dict['test_idx'],
                'labels': split_dict['test_labels'],
                'num_classes': split_dict['num_classes']
            }
        else:
            raise ValueError(f"Wrong task name: {params['task_name']}")
    
    full_data = get_data(params)

    all_acc_list = []
    all_microf_list = []
    all_macrof_list = []
    for task_dim, fewshot_tasks in fewshot_tasks_dict.items():
        time_start = time.time()
        test_task = test_split_dict[task_dim]
        num_classes = test_task["num_classes"]
        params["num_classes"] = test_task["num_classes"]
        print(f"{params['task_name']} Classification on {params['finetune_dataset']} "
            f"Y-Dim({task_dim}) with {num_classes}-class {params['n_shot']}-shot")
        
        acc_lst = []
        macrof_lst = []
        microf_lst = []
        for idx, train_task in enumerate(fewshot_tasks, start=1):
            encoder = get_unigraph2_encoder(ckpt_path, params['input_dim'], params['hidden_dim'], device=device)
            
            task_time_start = time.time()
            proto_emb, proto_classes = run_model_graph(encoder, full_data, train_task, params, is_train=True, proto_emb=None, proto_classes=None, is_batch=is_batch)
            with torch.no_grad():
                preds, test_labels = run_model_graph(encoder, full_data, test_task, params, False, proto_emb, proto_classes, is_batch)

            acc = torch.sum(preds == test_labels).float() / test_labels.size(0)
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_labels.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="micro")
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="macro")
            microf_lst.append(micro_f1 * 100)
            macrof_lst.append(macro_f1 * 100)
            acc_lst.append(acc * 100)
            task_time_end = time.time()

            print("=" * 10)
            print(f"Task {idx} Test Acc: {acc * 100:.2f}%\n"
                f"Task {idx} Micro-F1: {micro_f1 * 100:.2f}%\n"
                f"Task {idx} Macro-F1: {macro_f1 * 100:.2f}%")

        acc_tensor = torch.stack(acc_lst)
        acc_mean = acc_tensor.mean().item()
        microf_mean = sum(microf_lst) / len(microf_lst)
        macrof_mean = sum(macrof_lst) / len(macrof_lst)
        acc_std = acc_tensor.std().item()
        microf_std = torch.std(torch.tensor(microf_lst)).item()
        macrof_std = torch.std(torch.tensor(macrof_lst)).item()
        print(f'===Y-Dim({task_dim}){"=" * 50}\n'
            f"Accuracy:[{acc_mean:.4f}±{acc_std:.4f}]\n"
            f"Micro-F1:[{microf_mean:.4f}±{microf_std:.4f}]\n"
            f"Macro-F1:[{macrof_mean:.4f}±{macrof_std:.4f}]\n"
            f"Total Time: {time.time() - time_start:.2f}s")

        all_acc_list.extend(acc_lst)
        all_microf_list.extend(microf_lst)
        all_macrof_list.extend(macrof_lst)

    if len(fewshot_tasks_dict) > 1:
        all_acc_tensor = torch.stack(all_acc_list)
        all_acc_mean = all_acc_tensor.mean().item()
        all_microf_mean = sum(all_microf_list) / len(all_microf_list)
        all_macrof_mean = sum(all_macrof_list) / len(all_macrof_list)
        all_acc_std = all_acc_tensor.std().item()
        all_microf_std = torch.std(torch.tensor(all_microf_list)).item()
        all_macrof_std = torch.std(torch.tensor(all_macrof_list)).item()
        print(f'===Overall{"=" * 50}\n'
            f"Accuracy:[{all_acc_mean:.4f}±{all_acc_std:.4f}]\n"
            f"Micro-F1:[{all_microf_mean:.4f}±{all_microf_std:.4f}]\n"
            f"Macro-F1:[{all_macrof_mean:.4f}±{all_macrof_std:.4f}]")


if __name__ == "__main__":
    params = get_args_finetune()
    print("=" * 10)
    print(f"begin fine-tune: {params['finetune_dataset']}")
    print("=" * 10)
    main(params)
