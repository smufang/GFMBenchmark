import numpy as np
import torch.nn.functional as F
import torch
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import f1_score, roc_auc_score

task2metric = {'node': 'acc', 'link': 'acc', 'graph': 'auc'}


def evaluate(pred, y, mask=None, params=None):
    if params is None:
        params = {}
    metric = task2metric.get(params.get('task', 'node'), 'acc')

    if metric == 'acc':
        return eval_acc(pred, y, mask) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")
        

def eval_acc(y_pred, y_true, mask):
    # y_pred: logits (N, C)
    # y_true: (N,) labels
    device = y_pred.device
    num_classes = y_pred.size(1) if y_pred.ndim == 2 else int(y_pred.max().item() + 1)

    try:
        evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        if mask is not None:
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                return evaluator(y_pred[mask], y_true[mask]).item()
            else:
                idx = torch.tensor(mask, dtype=torch.long).to(device)
                return evaluator(y_pred[idx], y_true[idx]).item()
        else:
            return evaluator(y_pred, y_true).item()
    except Exception as e:
        try:
            pred_labels = y_pred.argmax(dim=1)
            correct = (pred_labels == y_true).float()
            return correct.mean().item()
        except Exception:
            print(f"eval_acc error: {e}, return 0.0")
            return 0.0


def eval_auc(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))

    if len(roc_list) == 0:
        return 0.0
    return sum(roc_list) / len(roc_list)
