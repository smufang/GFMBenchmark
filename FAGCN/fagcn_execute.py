import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.FAGCN.model import FAGCN
import dgl
from dgl import DGLGraph
from utils import process
from sklearn.metrics import f1_score
import csv
import torch_geometric
from torch_geometric.datasets import WebKB
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.datasets import HeterophilousGraphDataset
from models.FAGCN.utils import accuracy, preprocess_data, normalize_features
import argparse
import tqdm
from torch_geometric.loader import DataLoader
import utils.aug as aug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="wisconsin", help='data')
parser.add_argument('--drop_percent', type=float, default=0.05, help='drop percent')
parser.add_argument('--drop_edge', type=float, default=0.2, help='drop percent for edge dropping')
parser.add_argument('--dropout', type=float, default=0, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--use_origin_feature',default=False,   action="store_true", help='aug type: mask or edge')
# pretraining hyperperemeter
parser.add_argument('--num_layers', type=int, default=2, help='num of layers')
parser.add_argument('--down_lr', type=float, default=0.01, help='lr for downstream')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=20, help='patience')
parser.add_argument('--ds_epochs', type=int, default=1000, help='number of downstream epochs')
# downstream hyperperemeter
parser.add_argument('--out_size', type=int, default=256, help='hidden size of metanet output')
parser.add_argument("--n_shot", type=int, default=1)
parser.add_argument('--hid_units', type=int, default=256, help='hidden size')
parser.add_argument("--n_task", type=int, default=50)
args = parser.parse_args()

print('-' * 10)
print(args)
print('-' * 10)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

datasets = wisconsin()
features, adj, labels, _, _ = process.process_webkb(datasets.data, datasets.data.x.shape[0])
features = features.cuda()

labels = torch.FloatTensor(labels[np.newaxis]).cuda()

ft_size = features.shape[-1]  # node features dim
device="cuda"

g_origin, nclass, features_origin, labels_origin = preprocess_data(datasets)
#g_origin = g_origin.to(features_origin.device)
g_origin = g_origin.to(features.device)
deg = g_origin.in_degrees().cuda().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g_origin.ndata['d'] = norm

if args.use_origin_feature ==True:
    features_origin = features.cuda()
else:
    features_origin = features_origin.cuda()


'''------------------------------------------------------------
Downstream
------------------------------------------------------------'''
task_name = "node"
dataset_name = "Wisconsin"
fewshot_base_dir = os.path.join(BASE_DIR, "..", "datasets_split", dataset_name, "few-shot")
fewshot_tasks_dict = {}
for folder in os.listdir(fewshot_base_dir):
    if folder.startswith(f"{task_name}-"):
        task_dim = int(folder.split('-')[-1])
        folder_path = os.path.join(fewshot_base_dir, folder)
        tasks = []
        for file in os.listdir(folder_path):
            if file.endswith(".pt") and f"_{args.n_shot}-shot_{args.n_task}-tasks" in file:
                tasks += torch.load(os.path.join(folder_path, file))
        fewshot_tasks_dict[task_dim] = tasks
    else:
        raise ValueError(f"Wrong task name: {task_name}")

split_dir = os.path.join(".", "datasets_split", dataset_name, "split")
if not os.path.exists(split_dir):
    raise FileNotFoundError(f"No split directory found: {split_dir}")
test_split_dict = {}
for file in os.listdir(split_dir):
    if file.startswith(f"split_{task_name}_") and file.endswith(".pt"):
        task_dim = int(file.split('_')[-1].split('.')[0])
        split_dict = torch.load(os.path.join(split_dir, file))
        test_split_dict[task_dim] = {
            'idx': split_dict['test_idx'],
            'labels': split_dict['test_labels'],
            'num_classes': split_dict['num_classes']
        }
    else:
        raise ValueError(f"Wrong task name: {task_name}")

all_acc_list = []
all_microf_list = []
all_macrof_list = []
for task_dim, fewshot_tasks in fewshot_tasks_dict.items():
    time_start = time.time()
    test_task = test_split_dict[task_dim]
    num_classes = test_task["num_classes"]
    print(f"{task_name} Classification on {dataset_name} "
        f"Y-Dim({task_dim}) with {num_classes}-class {args.n_shot}-shot")
    
    acc_lst = []
    macrof_lst = []
    microf_lst = []
    total_epochs = 0
    for idx, train_task in enumerate(fewshot_tasks, start=1):
        xent = nn.CrossEntropyLoss()

        idx_train = torch.tensor(train_task["idx"]).cuda()
        train_lbls = torch.tensor(train_task["labels"]).cuda()

        log = FAGCN(g_origin, ft_size, args.hid_units, args.hid_units, args.drop_percent, args.eps, args.num_layers).cuda()
        opt = torch.optim.Adam(log.parameters(), lr=args.down_lr, weight_decay=args.weight_decay)
        classifier = torch.nn.Linear(args.hid_units, num_classes).cuda()

        best = 1e9
        cnt_wait = 0
        for epoch in range(args.ds_epochs):
            log.train()
            opt.zero_grad()
            emb = log.forward(features).float().cuda()
            logits = classifier(emb)
            loss = xent(logits[idx_train], train_lbls)
            if not loss.requires_grad:
                loss.requires_grad = True

            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            loss.backward(retain_graph=True)
            opt.step()

        log.eval()
        idx_test = torch.tensor(test_task["idx"]).cuda()
        test_lbls = torch.tensor(test_task["labels"]).cuda()
        emb = log.forward(features).float().cuda()
        logits = classifier(emb)

        preds = logits[idx_test].argmax(dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        preds_cpu = preds.cpu().numpy()
        test_lbls_cpu = test_lbls.cpu().numpy()
        micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="micro")
        macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="macro")
        microf_lst.append(micro_f1 * 100)
        macrof_lst.append(macro_f1 * 100)
        acc_lst.append(acc * 100)
        task_time_end = time.time()

        print('=' * 10)
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