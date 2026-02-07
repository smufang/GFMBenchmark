import os
import sys
import argparse
import random
import time
import datetime
from unittest import loader
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, HeteroData, TemporalData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torchmetrics import Accuracy
from utils.subbatch_loader import BatchGraphLoader
from sklearn.metrics import f1_score
from preprompt import PrePrompt
from downprompt import downprompt
from utils.others import create_x, complete_data
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils import compress_func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

def get_args_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "--dataset", "--finetune_dataset", type=str, default="Cora")
    parser.add_argument('--seed', type=int, default=39, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--down_type', type=str, default='Graphprompt')
    parser.add_argument('--usemlp', type=str, default='no', help='yes(DGI downstream) or no(Graphprompt downstream)')
    parser.add_argument('--premodel', type=str, default='DGI', help='the type of pretrain model')
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument('--input_dim', type=int, default=50)
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--compress_function', type=str, default='pca',
                            help='dimension alignment method: pca/svd')
    parser.add_argument("--task_name", type=str, default="node") #edge, graph
    parser.add_argument("--n_task", type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
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
    data_name = params['data']
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
        data.edge_attr = compress_func(data.edge_attr, k=params["input_dim"])
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


def run_model(model, log, optimizer, full_data, task, params, is_train):
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
    num_classes = params["num_classes"]

    loader = get_loader(params, data, task_labels, task_ids, is_train)

    if is_train:
        xent = nn.CrossEntropyLoss()
        loss = 0
        for batch, batch_labels, batch_ids in loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            y = batch_labels.to(device)
            batch_ids = batch_ids.to(device)
            node_feat = getattr(batch, 'x', None).to(device)
            edge_index = getattr(batch, 'edge_index', None).to(device)

            log.support(node_feat, edge_index, model.gcn, batch_ids, batch.batch, y, params['task_name'])
            logits = log.predict(node_feat, edge_index, model.gcn, batch_ids, batch.batch, params['task_name']).float()

            batch_loss = xent(logits, y.float())
            loss += batch_loss.item()

            optimizer.step()
            del batch, batch_labels, batch_ids, node_feat, edge_index, y, logits
            torch.cuda.empty_cache()

        return loss
    else:
        pred_list, y_list = [], []
        for batch, batch_labels, batch_ids in loader:
            batch = batch.to(device)
            y = batch_labels.to(device)
            batch_ids = batch_ids.to(device)
            node_feat = getattr(batch, 'x', None).to(device)
            edge_index = getattr(batch, 'edge_index', None).to(device)

            logits = log.predict(node_feat, edge_index, model.gcn, batch_ids, batch.batch, params['task_name']).float()

            pred_list.append(logits)
            y_list.append(y)
        
        del batch, batch_labels, batch_ids, node_feat, edge_index, y
        torch.cuda.empty_cache()
        
        preds = torch.cat(pred_list, dim=0)
        labels = torch.cat(y_list, dim=0)
        return preds, labels


def run(params):
    save_name = os.path.join(BASE_DIR, "checkpoints", f'id{params["input_dim"]}_hd{params["hid_units"]}_{params["premodel"]}_{params["data"]}.pt')
    
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    fewshot_base_dir = os.path.join(BASE_DIR, "..", "datasets_split", params['data'], "few-shot")
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

    split_dir = os.path.join(".", "datasets_split", params['data'], "split")
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
        print(f"{params['task_name']} Classification on {params['data']} "
            f"Y-Dim({task_dim}) with {num_classes}-class {params['n_shot']}-shot")
        
        acc_lst = []
        macrof_lst = []
        microf_lst = []
        total_epochs = 0
        for idx, train_task in enumerate(fewshot_tasks, start=1):
            model = PrePrompt(params["input_dim"], params["hid_units"], num_layers_num=3, dropout=0.1, premodel=params["premodel"])

            model.load_state_dict(torch.load(save_name))
            model = model.to(device)
            print(f"load pretrain model: {save_name}")

            log = downprompt(params["hid_units"], 8, params["input_dim"], params["num_classes"], think_layer_num=4,condition_layer_num=1, type=params["down_type"],usemlp=params["usemlp"]).to(device)
            for name, param in log.named_parameters():
                param.requires_grad = True
            log.train()

            optimizer = torch.optim.Adam([{'params': log.parameters()}], lr=0.001)
            
            cnt_wait = 0
            best_loss = 1e9
            task_time_start = time.time()
            for epoch in range(params['epochs']):
                loss = run_model(model, log, optimizer, full_data, train_task, params, is_train=True)
                
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), save_name)
                else:
                    cnt_wait += 1

                if cnt_wait == params["patience"]:
                    print('Early stopping!')
                    break

            log.eval()
            preds, test_labels = run_model(model, log, optimizer, full_data, test_task, params, is_train=False)
            acc = torch.sum(preds == test_labels).float() / test_labels.size(0)
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_labels.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="micro")
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="macro")
            microf_lst.append(micro_f1 * 100)
            macrof_lst.append(macro_f1 * 100)
            acc_lst.append(acc * 100)
            task_time_end = time.time()

            print(f"Task {idx} Test Acc: {acc * 100:.2f}%\n"
                f"Task {idx} Micro-F1: {micro_f1 * 100:.2f}%\n"
                f"Task {idx} Macro-F1: {macro_f1 * 100:.2f}%")
            
            total_epochs += epoch + 1

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
            f"Total Time: {time.time() - time_start:.2f}s\n"
            f"Total Epochs: {total_epochs}\n"
            f"Average Time per epoch: {(time.time() - time_start)/total_epochs:.4f}s")

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
    print(f"begin fine-tune")
    print("=" * 10)

    run(params)