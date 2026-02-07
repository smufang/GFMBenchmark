import os
import sys
from copy import deepcopy
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData, TemporalData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torchmetrics import Accuracy
from utils.subbatch_loader import BatchGraphLoader
from torch.optim import AdamW
from torch.nn import ReLU
from sklearn.metrics import f1_score
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.early_stop import TrainLossEarlyStopping
from utils.args import get_args_finetune
from utils.others import create_x, complete_data
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from utils import compress_func
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from data_provider.data_generator import *

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
    ### exp1 graph
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


def run_model_graph(model, full_data, task, params, is_train, proto_emb, is_batch, cache):
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
    data.edge_task_ids = task_ids
    if cache is None:
        if not is_batch:
            if params['task_name'] == 'edge':
                task_ids = data.edge_index[:, task_ids]
            cache = [(data, task_labels, task_ids)]
        else:
            loader = get_loader(params, data, task_labels, task_ids, is_train)
            cache = list(loader)
        
    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    num_classes = params["num_classes"]

    if is_train:
        proto_loss = torch.tensor(0.0).to(device)
        act_loss = torch.tensor(0.0).to(device)
        code_list, y_list = [], []
        if use_proto_clf:
            for batch, batch_labels, batch_ids in cache:
                try:
                    batch = batch.to(device)
                    batch.name = [params['finetune_dataset']] * batch.num_nodes
                    batch_ids = batch_ids.to(device)
                    y = batch_labels.to(device)

                    node_feat = getattr(batch, 'x', None).to(device)
                    edge_index = getattr(batch, 'edge_index', None).to(device)
                    edge_attr = getattr(batch, 'edge_attr', None)
                    if edge_attr is None:
                        edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float).to(device)
                    else:
                        edge_attr = edge_attr.to(device)

                    if params['task_name'] == 'graph':
                        z = model.encode_graph(node_feat, edge_index, edge_attr, batch.batch, pool="mean")
                    else:
                        z = model.encode(node_feat, edge_index, edge_attr)

                    code, _ = model.get_codes(z, use_orig_codes=True)
                    support_z, support_code = z[batch_ids], code[batch_ids]

                    code_list.append(support_code.detach())
                    y_list.append(y)

                except Exception as e:
                    if not is_batch:
                        print(f"{e}\n Switching to batch mode.")
                        is_batch = True
                        return run_model_graph(model, full_data, task, params, is_train, proto_emb, is_batch, cache)
                    else:
                        print(f"Num Nodes: {batch.num_nodes}")
                        print(f"Num Edges: {batch.num_edges}")
                        raise e

        if use_proto_clf:
            if params['task_name'] == 'edge':
                code = torch.cat(code_list, dim=1)
                code = code.mean(dim=0)
            else:
                code = torch.cat(code_list, dim=0)
            y = torch.cat(y_list, dim=0)
            proto_emb = model.get_class_prototypes(code, y, num_classes)
    
        total_proto_loss = 0
        total_act_loss = 0
        total_loss = 0
        for batch, batch_labels, batch_ids in cache:
            batch = batch.to(device)
            y = batch_labels.to(device)
            batch_ids = batch_ids.to(device)
            batch.name = [params['finetune_dataset']] * batch.num_nodes
            
            node_feat = getattr(batch, 'x', None).to(device)
            edge_index = getattr(batch, 'edge_index', None).to(device)
            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float).to(device)
            else:
                edge_attr = edge_attr.to(device)
            
            if params['task_name'] == 'graph':
                z = model.encode_graph(node_feat, edge_index, edge_attr, batch.batch, pool="mean")
            else:
                z = model.encode(node_feat, edge_index, edge_attr)
            
            if use_proto_clf:
                code, _ = model.get_codes(z, use_orig_codes=True)
                support_z = z[batch_ids]
                if params['task_name'] == 'edge':
                    support_z = support_z.mean(dim=0)
                
                query_emb = support_z
                if params['task_name'] == 'graph':
                    proto_loss = model.compute_proto_loss(query_emb, proto_emb, y, task="single") * params["lambda_proto"]
                else:
                    proto_loss = model.compute_proto_loss(query_emb, proto_emb, y) * params["lambda_proto"]

            if use_lin_clf:
                if params['task_name'] == 'graph':
                    act_loss = model.compute_activation_loss(support_z, y, task="single") * params["lambda_act"]
                else:
                    act_loss = model.compute_activation_loss(support_z, y) * params["lambda_act"]

            loss = proto_loss + act_loss
            
            total_proto_loss += proto_loss.item()
            total_act_loss += act_loss.item()
            total_loss += loss.item()

            loss.backward()
            del batch, batch_labels, batch_ids, node_feat, edge_index, edge_attr, z, loss
            if use_proto_clf:
                del code, support_z, query_emb, proto_loss
            if use_lin_clf:
                del act_loss
            torch.cuda.empty_cache()

        return {
            'proto_loss': total_proto_loss / task_labels.size(0),
            'act_loss': total_act_loss / task_labels.size(0),
            'loss': total_loss / task_labels.size(0),
        }, proto_emb, cache

    else: #test
        pred_proto = 0
        pred_lin = 0
        pred_list, y_list = [], []
        with torch.no_grad():
            for batch, batch_labels, batch_ids in cache:
                try:
                    batch = batch.to(device)
                    y = batch_labels.to(device)
                    batch_ids = batch_ids.to(device)
                    batch.name = [params['finetune_dataset']] * batch.num_nodes
                    
                    node_feat = getattr(batch, 'x', None).to(device)
                    edge_index = getattr(batch, 'edge_index', None).to(device)
                    edge_attr = getattr(batch, 'edge_attr', None)

                    if edge_attr is None:
                        edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float).to(device)
                    else:
                        edge_attr = edge_attr.to(device)
                    
                    if params['task_name'] == 'graph':
                        z = model.encode_graph(node_feat, edge_index, edge_attr, batch.batch, pool="mean")
                    else:
                        z = model.encode(node_feat, edge_index, edge_attr)
                    
                    if use_proto_clf:
                        test_z = z[batch_ids]
                        if params['task_name'] == 'edge':
                            test_z = test_z.mean(dim=0)

                        query_emb = test_z
                        if params['task_name'] == 'graph':
                            pred_proto = model.get_proto_logits(query_emb, proto_emb, task="single")
                        else:
                            pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        if params['task_name'] == 'graph':
                            pred_lin = model.get_lin_logits(test_z).mean(1)
                        else:
                            pred_lin = model.get_lin_logits(test_z).mean(1).softmax(dim=-1)
                    
                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin
                    if params['task_name'] == 'graph':
                        pred_list.append(pred.detach())
                        y_list.append(y)
                    else:
                        pred = torch.argmax(pred, dim=1)
                        pred_list.append(pred.detach())
                        y_list.append(y)

                except Exception as e:
                    if not is_batch:
                        print(f"{e}\n Switching to batch mode.")
                        is_batch = True
                        return run_model_graph(model, full_data, task, params, is_train, proto_emb, is_batch, cache)
                    else:
                        print(f"Num Nodes: {batch.num_nodes}")
                        print(f"Num Edges: {batch.num_edges}")
                        raise e

                del batch, batch_labels, batch_ids, node_feat, edge_index, edge_attr, z, pred
                if use_proto_clf:
                    del query_emb, pred_proto, test_z
                if use_lin_clf:
                    del pred_lin
                torch.cuda.empty_cache()
        
        preds = torch.cat(pred_list, dim=0)
        labels = torch.cat(y_list, dim=0)
        return preds, labels, cache


def run(params):
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    is_batch = True
    fewshot_base_dir = os.path.join(BASE_DIR, "..", "..", "datasets_split", params['finetune_dataset'], "few-shot")
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
        total_epochs = 0
        test_cache = None
        for idx, train_task in enumerate(fewshot_tasks, start=1):
            encoder = Encoder(
                input_dim=params["input_dim"],
                hidden_dim=params["hidden_dim"],
                activation=ReLU,
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
            )

            path = os.path.join(
                params['pt_model_path'],
                f"input_dim{params['input_dim']}_hd{params['hidden_dim']}_codebook_size_{params['codebook_size']}_layer_{params['num_layers']}_"
                f"pretrain_on_exp{params['exp']}_seed_{params['pretrain_seed']}"
            )
            encoder_state = torch.load(os.path.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt'))
            vq_state = torch.load(os.path.join(path, f'vq_{params["pretrain_model_epoch"]}.pt'))
            encoder.load_state_dict(encoder_state, strict=False)
            vq.load_state_dict(vq_state, strict=False)
            encoder = encoder.to(device)
            vq = vq.to(device)

            print(f"load pretrain model: {path}(strict=False)")

            model = TaskModel(encoder=deepcopy(encoder), vq=deepcopy(vq), num_classes=num_classes, params=params).to(device)
            optimizer = AdamW(model.parameters(), lr=params.get("finetune_lr", 0.001))
            stopper = TrainLossEarlyStopping(patience=params.get("early_stop", 10))
            model.train()

            task_time_start = time.time()
            train_cache = None
            proto_emb = None
            for epoch in range(params['finetune_epochs']):
                optimizer.zero_grad()
                loss_dict, proto_emb, train_cache = run_model_graph(model, full_data, train_task, params, True, proto_emb, is_batch, train_cache)
                train_loss = loss_dict["loss"]
                optimizer.step()
                stop = stopper(train_loss, model)
                if stop:
                    print(f"Early stopped at epoch {epoch} (task {idx})")
                    break
            
            print(f"Task {idx} Training End, best loss={stopper.best_loss:.4f}")
            model.load_state_dict(stopper.best_model_state)
            model.eval()
            with torch.no_grad():
                preds, test_labels, test_cache = run_model_graph(model, full_data, test_task, params, False, proto_emb, is_batch, test_cache)

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
    print(params)
    params['pt_model_path'] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'ckpts', 'pretrain_model')
    )
    print("=" * 10)
    print(f"begin fine-tune: {params['finetune_dataset']}")
    print("=" * 10)

    run(params)