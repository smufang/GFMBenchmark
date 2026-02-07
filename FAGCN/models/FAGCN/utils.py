import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *

def preprocess_data(dataset):
    data = dataset.data
    features = normalize_features(data.x.float())
    features = torch.FloatTensor(features)

    labels = data.y.long().cuda()
    nclass = int(labels.max().item() + 1)

    src, dst = data.edge_index
    g_origin = dgl.graph((src, dst), num_nodes=data.num_nodes)

    g_origin = dgl.to_simple(g_origin)
    g_origin = dgl.to_bidirected(g_origin)
    g_origin = dgl.remove_self_loop(g_origin)

    return g_origin, nclass, features, labels


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


