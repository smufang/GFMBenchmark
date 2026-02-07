import numpy as np
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from torch_scatter import scatter_mean

from model.vq import l2norm


# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def distance_metric(a, b, use_cosine_sim=True):
    # a shape: [n, d]
    # b shape: [m, d]

    if use_cosine_sim:
        a = l2norm(a)
        b = l2norm(b)

        cross_term = torch.mm(a, b.t())
        logits = 2 - 2 * cross_term
    else:
        a_sq = torch.sum(a ** 2, dim=1).unsqueeze(1)  # Shape: [n, 1]
        b_sq = torch.sum(b ** 2, dim=1).unsqueeze(0)  # Shape: [1, m]
        cross_term = torch.mm(a, b.t())  # Shape: [n, m]

        logits = a_sq + b_sq - 2 * cross_term

    return -logits


def efficient_compute_class_prototypes(embeddings, classes, num_classes_in_total, return_head_first=True):
    # Embeddings (z) shape: [n, d] or [n, h, d] or [r, n, h, d]
    # Classes shape: [n] or [r, n]
    # return_head_first: if True, the first dimension of the output will be the heads, otherwise it will be the classes

    embeddings = l2norm(embeddings)

    ndim = embeddings.ndim
    assert ndim in [2, 3, 4]

    if ndim == 4:
        num_runs = embeddings.shape[0]
    else:
        num_runs = 1

    # Rearrange the embeddings as [run, head, num_nodes, dim]
    # classes as [run, num_nodes]
    if ndim == 2:
        embeddings = rearrange(embeddings, "n d -> 1 1 n d")
        classes = rearrange(classes, "n -> 1 n")
    elif ndim == 3:
        embeddings = rearrange(embeddings, "n h d -> 1 h n d")
        classes = rearrange(classes, "n -> 1 n")
    elif ndim == 4:
        embeddings = rearrange(embeddings, "r n h d -> r h n d")

    # Compute the class prototypes for each run.
    class_prototypes = []
    for i in range(num_runs):
        class_prototypes.append(
            scatter_mean(
                embeddings[i], classes[i], dim=1, dim_size=num_classes_in_total
            )
        )
    class_prototypes = torch.stack(class_prototypes, dim=0)  # [r, h, c, d]

    if ndim == 2:
        class_prototypes = rearrange(class_prototypes, "1 1 c d -> c d")
    elif ndim == 3:
        class_prototypes = rearrange(class_prototypes, "1 h c d -> h c d")

    if return_head_first:
        return class_prototypes
    else:
        if ndim == 3:
            return rearrange(class_prototypes, "h c d -> c h d")
        elif ndim == 4:
            return rearrange(class_prototypes, "r h c d -> r c h d")


def compute_multitask_loss(pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


class TaskModel(nn.Module):
    def __init__(self, encoder, vq, num_classes, params):
        super().__init__()

        self.encoder = encoder
        self.vq = vq

        num_heads, codebook_size, code_dim = vq.codebook.shape
        self.num_classes = num_classes
        self.num_heads = vq._codebook.num_codebooks
        self.code_dim = code_dim

        self.separate_decoder_for_each_head = params["separate_decoder_for_each_head"]
        self.use_z_in_predict = params["use_z_in_predict"]
        self.use_cosine_sim = params["use_cosine_sim"]
        self.lambda_proto = params["lambda_proto"]
        self.lambda_act = params["lambda_act"]
        self.trade_off = params["trade_off"]
        self.num_instances_per_class = params["num_instances_per_class"]

        if self.separate_decoder_for_each_head:
            self.decoder = nn.Linear(code_dim * num_heads, num_classes * num_heads)
        else:
            self.decoder = nn.Linear(code_dim, num_classes)

    def encode(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        return z

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None, pool="mean"):
        z = self.encoder(x, edge_index, edge_attr)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def get_codes(self, z, use_orig_codes=True):
        quantize, indices, commit_loss, codes = self.vq(z)
        if use_orig_codes:
            return codes.reshape(-1, self.num_heads, self.code_dim), commit_loss
        else:
            return quantize, commit_loss

    def get_class_prototypes(self, z, y, num_classes_in_total):
        if isinstance(y, dict):
            # This works for graph classification with multiple binary tasks

            n_task = len(y)
            flat_y = np.array([])

            for task, labels in y.items():
                flat_y = np.concatenate((flat_y, task * 2 + labels), axis=0)
            flat_y = torch.tensor(flat_y, dtype=torch.long, device=z.device)

            proto_emb = efficient_compute_class_prototypes(
                z, flat_y, num_classes_in_total * 2, return_head_first=False
            )
            return proto_emb.resize(n_task, 2, self.num_heads, self.code_dim)

        else:
            # This works for node and link classification
            return efficient_compute_class_prototypes(
                z, y, num_classes_in_total, return_head_first=False
            )

    def compute_proto_loss(self, query_emb, proto_emb, y, task="single"):
        # query_emb in [n, d] or [n, h, d]
        # proto_emb in [c, d] or [c, h, d]
        ndim_query = query_emb.ndim
        ndim_proto = proto_emb.ndim

        assert ndim_query in [2, 3]
        assert ndim_proto in [2, 3, 4]

        if ndim_query == 2:
            query_emb = rearrange(query_emb, "n d -> n 1 d")
        if ndim_proto == 2:
            proto_emb = rearrange(proto_emb, "c d -> c 1 d")
        if ndim_proto == 4 and task == 'multi':
            # This works for multitask learning (binary)
            n_task = proto_emb.shape[0]
            proto_emb = rearrange(proto_emb, "t c h d -> (t c) h d")

        query_emb = rearrange(query_emb, "n h d -> h n d")
        proto_emb = rearrange(proto_emb, "c h d -> h c d")

        query_heads = query_emb.shape[0]
        proto_heads = proto_emb.shape[0]
        num_heads = max(query_heads, proto_heads)

        proto_loss = 0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter, self.use_cosine_sim)

            if task == "single":
                proto_loss += F.cross_entropy(logits, y)
            elif task == "multi":
                logits = rearrange(logits, "n (t c) -> n t c", t=n_task, c=2)
                logits = logits[:, :, 0] - logits[:, :, 1]  # The 0-th is positive, the 1-th is negative
                proto_loss += compute_multitask_loss(logits, y)
            else:
                raise ValueError('task must be either "single" or "multi"')
        proto_loss /= num_heads

        return proto_loss

    def compute_proto_reg(self, proto_emb):
        # proto_emb in [c, d] or [c, h, d]
        ndim = proto_emb.ndim
        if ndim == 2:
            return 0
        if ndim == 4:
            proto_emb = rearrange(proto_emb, "t c h d -> (t c) h d")

        proto_emb = rearrange(proto_emb, "c h d -> h c d")
        proto_mean = proto_emb.mean(0)

        num_heads = proto_emb.shape[0]

        proto_reg = 0
        for h in range(num_heads):
            proto_reg += F.kl_div(
                proto_emb[h].log_softmax(dim=-1),
                proto_mean.softmax(dim=-1),
                reduction="batchmean",
            )
        proto_reg /= num_heads

        return proto_reg

    def compute_activation_loss(self, z, y, task="single"):
        if task == "single":
            pred = self.get_lin_logits(z).mean(1)
            return F.cross_entropy(pred, y)
        elif task == "multi":
            pred = self.get_lin_logits(z).mean(1)
            return compute_multitask_loss(pred, y)
        else:
            raise ValueError('task must be either "single" or "multi"')

    def get_lin_logits(self, z):
        quantize, indices, commit_loss, codes = self.vq(z)
        if self.separate_decoder_for_each_head:
            pred = self.decoder(codes).reshape(-1, self.num_heads, self.num_classes)
        else:
            pred = self.decoder(quantize).reshape(-1, 1, self.num_classes)
        return pred

    def get_proto_logits(self, query_emb, proto_emb, task='single'):
        # query_emb in [n, d] or [n, h, d]
        # proto_emb in [c, d] or [c, h, d]
        ndim_query = query_emb.ndim
        ndim_proto = proto_emb.ndim

        assert ndim_query in [2, 3]
        assert ndim_proto in [2, 3, 4]

        if ndim_query == 2:
            query_emb = rearrange(query_emb, "n d -> n 1 d")
        if ndim_proto == 2:
            proto_emb = rearrange(proto_emb, "c d -> c 1 d")
        if ndim_proto == 4:
            n_task = proto_emb.shape[0]
            proto_emb = rearrange(proto_emb, "t c h d -> (t c) h d")

        query_emb = rearrange(query_emb, "n h d -> h n d")
        proto_emb = rearrange(proto_emb, "c h d -> h c d")

        query_heads = query_emb.shape[0]
        proto_heads = proto_emb.shape[0]
        num_heads = max(query_heads, proto_heads)

        total_logits = 0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter, self.use_cosine_sim)
            if task == 'multi':
                logits = rearrange(logits, "n (t c) -> n t c", t=n_task, c=2)
                logits = logits[:, :, 0] - logits[:, :, 1]  # The 0-th is positive, the 1-th is negative
            total_logits += logits

        total_logits = total_logits / num_heads

        return total_logits

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        out = self.get_lin_logits(z)
        return out


class TaskModelWithoutVQ(nn.Module):
    def __init__(self, encoder, vq, num_classes, params):
        super().__init__()

        assert vq is None

        self.encoder = encoder
        self.num_classes = num_classes

        self.lin = nn.Linear(params['hidden_dim'], num_classes)

    def encode(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        return z

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None, pool="mean"):
        z = self.encoder(x, edge_index, edge_attr)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def compute_loss(self, z, y, task="single"):
        pred = self.get_lin_logits(z)
        if task == "single":
            return F.cross_entropy(pred, y)
        elif task == "multi":
            return compute_multitask_loss(pred, y)
        else:
            raise ValueError('task must be either "single" or "multi"')

    def get_lin_logits(self, z):
        return self.lin(z)

    def get_logits(self, z):
        return self.lin(z)

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        logit = self.get_lin_logits(z)
        return logit, z
