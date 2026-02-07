import os
from os import path
import sys
import argparse
import numpy as np
import pandas as pd
import time
import datetime
import csv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.sparse as sp
import pickle as pkl
import sys
import torch_geometric.utils
import networkx as nx
import torch_geometric.transforms as T
from torch_scatter import scatter
from torch_geometric.utils import to_undirected, remove_isolated_nodes
torch.autograd.set_detect_anomaly(True)
from DSSL.dataset import NCDataset
from sklearn.metrics import f1_score

from DSSL.data_utils import sample_neighborhood, sample_neg_neighborhood
from DSSL.encoders import GCN, MLP
import faulthandler
faulthandler.enable()
# import process
from DSSL.models import DSSL
from torch_geometric.utils.convert import from_scipy_sparse_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
from data_provider.data_generator import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dataset', type=str, default='wisconsin')
parser.add_argument('--save_name', type=str, default='./modelset/dssl/wisconsin_DSSL.pkl', help='save ckpt name')
parser.add_argument('--sub_dataset', type=str, default='DE')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=int, default=0)# 0.01
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--directed', action='store_true', help='set to not symmetrize adjacency')
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=39, help='Random seed.')
parser.add_argument('--display_step', type=int, default=25, help='how often to print')
parser.add_argument('--train_prop', type=float, default=.48, help='training label proportion')
parser.add_argument('--valid_prop', type=float, default=.32, help='validation label proportion')
parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
parser.add_argument('--rand_split', type=bool, default=True, help='use random splits')
parser.add_argument('--embedding_dim', type=int, default=10, help="embedding dim")
parser.add_argument('--neighbor_max', type=int, default=5, help="neighbor num max")
parser.add_argument('--cluster_num', type=int, default=6, help="cluster num")
parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--entropy', type=float, default=0.0)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--encoder', type=str, default='MLP')
parser.add_argument('--mlp_bool', type=int, default=1, help="embedding with mlp predictor")
parser.add_argument('--tao', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--mlp_inference_bool', type=int, default=1, help="embedding with mlp predictor")
parser.add_argument('--neg_alpha', type=int, default=0, help="negative alpha ")
parser.add_argument("--n_shot", type=int, default=1)
parser.add_argument('--hid_units', type=int, default=64, help='number of neighbors')
parser.add_argument("--n_task", type=int, default=50)
parser.add_argument('--patience', type=float, default=20, help='patience')
args = parser.parse_args()
args = parser.parse_args()
print(args)

if args.lr == 0:
    args.lr = 0.001
elif args.lr == 1:
    args.lr = 0.01

### Seeds ###
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

### device ###
device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

### Load and preprocess data ###
def process_webkb(data, nb_nodes):
    nb_graphs = 1
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros((nb_nodes,5))
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    sizes = data.x.shape[0]
    features= data.x

    for i in range(data.y.shape[0]):
        labels[i][data.y[i]] = 1
    masks= 1.0
    e_ind = data.edge_index
    coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
    adjacency = coo.todense()
    adjacency = sp.csr_matrix(adjacency)
    return features, adjacency, labels, sizes, masks

datasets = wisconsin()
features, adj, labels, sizes, masks = process_webkb(datasets.data, datasets.data.x.shape[0])
edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
dataset = NCDataset(args.dataset)
edge_index = torch.tensor(edge_index, dtype=torch.long)
node_feat = torch.tensor(features, dtype=torch.float)
num_nodes = node_feat.size(0)
dataset.graph = {'edge_index': edge_index,
                    'node_feat': node_feat,
                    'edge_feat': None,
                    'num_nodes': num_nodes}
label = torch.tensor(labels, dtype=torch.long)
dataset.label = label

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                for _ in range(args.runs)]

dataset.graph['num_nodes']=dataset.graph['node_feat'].shape[0]
n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

if not args.directed:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

dataset.label = dataset.label.to(device)

sampled_neighborhoods = sample_neighborhood(dataset, device, args)
if args.neg_alpha:
    sampled_neg_neighborhoods = sample_neg_neighborhood(dataset, device, args)
    print('sample_neg_neighborhoods')

### Choose encoder ###
if args.encoder == 'GCN':
    encoder = GCN(in_channels=d,
                  hidden_channels=args.hidden_channels,
                  out_channels=args.hidden_channels,
                  num_layers=args.num_layers, use_bn=not args.no_bn,
                  dropout=args.dropout).to(device)
else:
    encoder = MLP(in_channels=d,
                  hidden_channels=args.hidden_channels,
                  out_channels=args.hidden_channels,
                  num_layers=args.num_layers,
                  dropout=args.dropout).to(device)

model = DSSL(encoder=encoder,
             hidden_channels=args.hidden_channels,
             dataset=dataset,
             device=device,
             cluster_num=args.cluster_num,
             alpha=args.alpha,
             gamma=args.gamma,
            tao=args.tao,
            beta=args.beta,
             moving_average_decay=args.tau).to(device)

if not args.mlp_bool: # 0 embedding without mlp predictor
    model.Embedding_mlp = False
if not args.mlp_inference_bool: # 0 embedding without mlp predictor
    model.inference_mlp = False


'''------------------------------------------------------------
Pretrain
------------------------------------------------------------'''
SAVE_DIR = os.path.join(BASE_DIR, args.save_name)
if not os.path.exists(SAVE_DIR):
    ## Training loop ###
    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float('-inf')
        loss_lst = []
        best_loss = float('inf')

        for epoch in range(args.epochs):
            # pre-training
            model.train()
            batch_size = args.batch_size
            perm = torch.randperm(n)
            epoch_loss = 0
            for batch in range(0, n, batch_size):
                optimizer.zero_grad()
                online_embedding = model.online_encoder(dataset)
                target_embedding = model.target_encoder(dataset)
                batch_idx = perm[batch:batch + batch_size]  # perm[2708,]
                batch_idx = batch_idx.to(device)
                batch_neighbor_index = sampled_neighborhoods[batch_idx].type(torch.long)
                batch_embedding = online_embedding[batch_idx].to(device)
                batch_embedding = F.normalize(batch_embedding, dim=-1, p=2)
                batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index.cpu()]
                batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                main_loss, context_loss, entropy_loss, k_node = model(batch_embedding, batch_neighbor_embedding)
                tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=args.cluster_num).type(torch.FloatTensor).to(
                    device)
                batch_sum = (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                if args.neg_alpha:
                    batch_neg_neighbor_index = sampled_neg_neighborhoods[batch_idx]
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neg_neighbor_index]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                    batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                    main_neg_loss, tmp, tmp, tmp = model(batch_embedding, batch_neighbor_embedding)
                    loss = main_loss + args.gamma * (context_loss + entropy_loss) + main_neg_loss

                else:
                    loss = main_loss+ args.gamma*(context_loss+entropy_loss)
                print("run : {}, batch : {}, main_loss: {}, context_loss: {}, entropy_loss: {}".format(run,batch,main_loss, context_loss, entropy_loss))
                loss.backward()
                optimizer.step()
                model.update_moving_average()
                epoch_loss = epoch_loss + loss
            if epoch %1== 0:
                model.eval()
                for batch in range(0, n, batch_size):
                    online_embedding = model.online_encoder(dataset).detach().cpu()
                    target_embedding = model.target_encoder(dataset).detach().cpu()
                    batch_idx = perm[batch:batch + batch_size]
                    batch_idx = batch_idx.to(device)
                    batch_neighbor_index = sampled_neighborhoods[batch_idx].type(torch.long)
                    batch_target_embedding = target_embedding[batch_idx.cpu()].to(device)
                    batch_embedding = online_embedding[batch_idx.cpu()].to(device)
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index.cpu()]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                    main_loss, context_loss, entropy_loss, k_node = model(batch_embedding, batch_neighbor_embedding)
                    tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=args.cluster_num).type(torch.FloatTensor).to(
                        device)
                    if batch == 0:
                        cluster = torch.matmul(batch_embedding.t(),tmp )
                        batch_sum=(torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                    else:
                        cluster+=torch.matmul(batch_embedding.t(),tmp)
                        batch_sum += (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                cluster = F.normalize(cluster, dim=-1, p=2)
                model.update_cluster(cluster,batch_sum)
            print("epoch: {}, loss: {}".format(epoch, epoch_loss))
            
    os.makedirs(os.path.join(BASE_DIR, 'modelset', 'dssl'), exist_ok=True)
    torch.save(model.state_dict(), SAVE_DIR)


'''------------------------------------------------------------
Downstream
------------------------------------------------------------'''
model.load_state_dict(torch.load(SAVE_DIR))
model.eval()
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
        log = nn.Linear(args.hid_units, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.001)
        xent = nn.CrossEntropyLoss()

        embedding = model.online_encoder(dataset)
        embedding = embedding.detach()

        idx_train = torch.tensor(train_task["idx"]).cuda()
        train_lbls = torch.tensor(train_task["labels"]).cuda()
        train_embedding = embedding[idx_train]

        best = 1e9
        cnt_wait = 0
        for epoch in range(args.epochs):
            log.train()
            opt.zero_grad()

            logits = log(train_embedding)
            #print("logits: ", logits.shape)
            #print("train_lbls: ", train_lbls.shape)
            loss = xent(logits, train_lbls)

            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            loss.backward()
            opt.step()
                
        log.eval()
        idx_test = torch.tensor(test_task["idx"]).cuda()
        test_lbls = torch.tensor(test_task["labels"]).cuda()
        test_embedding = embedding[idx_test]
        logits = log(test_embedding).float().cuda()
        preds = logits.argmax(dim=1)

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

