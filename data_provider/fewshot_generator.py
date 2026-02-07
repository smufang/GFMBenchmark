from utils.format_trans import hetero_to_data, temporal_to_data, multi_to_one
from torch_geometric.data import InMemoryDataset, Data, HeteroData, TemporalData
import torch
import random
from collections import defaultdict
from root import ROOT_DIR
import os


def process_multi_labels(labels, n=5):
    """
    For multi-label tasks with possible NaN values.

    Args:
        labels: Tensor of shape [num_nodes, num_dims], may contain NaNs
        n: number of top dimensions to select

    Returns:
        dict mapping dim_idx -> dict
            - dict: label (0 or 1) -> list of indices
    """

    def nanvar(tensor, dim):
        """Compute variance ignoring NaNs."""
        mean = torch.nanmean(tensor, dim=dim, keepdim=True)
        diff = tensor - mean
        diff_squared = diff ** 2
        return torch.nanmean(diff_squared, dim=dim)

    labels = labels.float()  # to calculate variance
    num_nodes, num_dims = labels.shape

    # Count NaNs per dimension
    nan_counts = torch.isnan(labels).sum(dim=0)
    # Compute variance ignoring NaNs
    var_per_dim = nanvar(labels, dim=0)

    # Sort dimensions: fewer NaNs first, then higher variance
    sort_order = sorted(
        range(num_dims),
        key=lambda i: (nan_counts[i].item(), -var_per_dim[i].item())
    )
    selected_dims = sort_order[:min(n, num_dims)]

    # Build label2idx dict per selected dimension
    label2idx_dict = {}
    for dim_idx in selected_dims:
        col = labels[:, dim_idx]
        valid_mask = ~torch.isnan(col)  # filter out NaNs
        bin_labels = (col[valid_mask] > 0).long()  # binarize

        label2idx = defaultdict(list)
        for i, y in zip(valid_mask.nonzero(as_tuple=False).view(-1).tolist(), bin_labels.tolist()):
            label2idx[y].append(i)

        label2idx_dict[dim_idx] = label2idx

    return label2idx_dict


def split_data(dataset: InMemoryDataset, name, task='node', train_ratio=0.2, K=5):
    """
    Generic function to split nodes, edges, or graphs into train/test sets by label.

    Args:
        dataset: PyG dataset object
                 - node/edge: use dataset[0]
                 - graph: use entire dataset
        name: dataset name (for saving)
        task: 'node', 'edge', 'graph'
        train_ratio: fraction of samples for training
        K: minimum samples per label in train set

    Returns:
        train_idx, test_idx, train_label2idx, test_labels
    """
    assert task in ['node', 'edge', 'graph'], "task must be 'node', 'edge', or 'graph'"
    # extract labels
    label2idx = defaultdict(list)
    if task in ['node', 'edge']:
        data = dataset[0]
        if isinstance(data, (HeteroData, TemporalData)):
            # use_embedding = True if name == 'ogbn-mag' else False
            data = hetero_to_data(data, need_y=True, use_embedding=False) if isinstance(data, HeteroData) \
                else temporal_to_data(data, need_y=True)
        assert isinstance(data, Data), "Data must be a PyG Data object after conversion"

        if task == 'node':
            labels = data.y.squeeze()
        else:
            labels = data.edge_type.squeeze()
    else:
        labels = dataset.data.y.squeeze()

    # process multi-labels
    if labels.dim() == 1:
        for i, y in enumerate(labels.tolist()):
            if y != -1:
                label2idx[y].append(i)
        label2idx_dict = {0: label2idx}
    else:
        label2idx_dict = process_multi_labels(labels)

    # split per label2idx
    for task_dim, label2idx in label2idx_dict.items():
        train_idx, test_idx = [], []
        train_label2idx = defaultdict(list)
        test_labels = []

        for y, ids in label2idx.items():
            random.shuffle(ids)
            n_train = max(int(len(ids) * train_ratio), K)
            train_idx.extend(ids[:n_train])
            train_label2idx[y] = ids[:n_train]

            test_idx.extend(ids[n_train:])
            test_labels.extend([y] * len(ids[n_train:]))

        save_path = f"{ROOT_DIR}/datasets_split/{name}/split/split_{task}_{task_dim}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_label2idx": dict(train_label2idx),
            "test_labels": test_labels,
            "num_classes": len(label2idx)
        }, save_path)
        print(f"Saved split ({task}) for {name} to {save_path}")


def generate_few_shot_tasks(train_label2idx, name, task='node', task_dim=0, N=5, K=5, num_tasks=50):
    """
    Generic function to generate N-way K-shot few-shot tasks.

    Args:
        train_label2idx: dict from split_data_generic
        name: dataset name
        task: 'node', 'edge', 'graph'
        task_dim: dimension index for multi-label tasks
        N: N-way
        K: K-shot
        num_tasks: number of tasks to generate

    Returns:
        List of tasks, each task is {'idx': [...], 'labels': [...]}
    """
    labels = list(train_label2idx.keys())
    tasks = []

    for _ in range(num_tasks):
        selected_labels = random.sample(labels, N)
        support_idx, support_labels = [], []
        for lbl in selected_labels:
            candidates = train_label2idx[lbl]
            sampled = random.choices(candidates, k=K)
            support_idx.extend(sampled)
            support_labels.extend([lbl] * K)
        tasks.append({"idx": support_idx, "labels": support_labels})

    save_path = f"{ROOT_DIR}/datasets_split/{name}/few-shot/{task}-{task_dim}/{N}-way_{K}-shot_{num_tasks}-tasks.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tasks, save_path)
    print(f"Saved {num_tasks} N-way K-shot {task} tasks to {save_path}")


def load_few_shot_tasks(name: str, task: str, K: int, num_tasks: int):
    """
    Load all few-shot tasks for all task_dims of a given dataset name and task type.
    """
    fewshot_base_dir = os.path.join(ROOT_DIR, "datasets_split", name, "few-shot")
    if not os.path.exists(fewshot_base_dir):
        raise FileNotFoundError(f"No few-shot tasks found in {fewshot_base_dir}.\n"
                                f"Run data_provider/fewshot_generator.py to generate them first.")

    fewshot_tasks_dict = {}
    # iterate all task-dim folders
    for folder in os.listdir(fewshot_base_dir):
        if folder.startswith(f"{task}-"):
            task_dim = int(folder.split('-')[-1])
            folder_path = os.path.join(fewshot_base_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".pt") and f"_{K}-shot_{num_tasks}-tasks" in file:
                    fewshot_tasks_dict[task_dim] = torch.load(os.path.join(folder_path, file))
    if not fewshot_tasks_dict:
        raise FileNotFoundError(f"No few-shot tasks found for dataset={name}, task={task}, K={K}, num_tasks={num_tasks}.")
    return fewshot_tasks_dict


def load_test_splits(name: str, task: str):
    """
    Load all test splits for all task_dims of a given dataset name and task type.
    """
    split_dir = os.path.join(ROOT_DIR, "datasets_split", name, "split")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"No split directory found: {split_dir}")

    test_split_dict = {}
    for file in os.listdir(split_dir):
        if file.startswith(f"split_{task}_") and file.endswith(".pt"):
            task_dim = int(file.split('_')[-1].split('.')[0])
            split_dict = torch.load(os.path.join(split_dir, file))
            test_split_dict[task_dim] = {
                'idx': split_dict['test_idx'],
                'labels': split_dict['test_labels'],
                'num_classes': split_dict['num_classes']
            }
    if not test_split_dict:
        raise FileNotFoundError(f"No test splits found for dataset={name}, task={task} in {split_dir}")
    return test_split_dict


if __name__ == '__main__':
    from data_provider import NC_exp1, EC_exp1, GC_exp1, NC_exp2, EC_exp2, GC_exp2

    all_generators = {
        **NC_exp1, **NC_exp2,
        **EC_exp1, **EC_exp2,
        **GC_exp1, **GC_exp2
    }

    node_datasets = set(NC_exp1.keys()) | set(NC_exp2.keys())
    edge_datasets = set(EC_exp1.keys()) | set(EC_exp2.keys())
    graph_datasets = set(GC_exp1.keys()) | set(GC_exp2.keys())

    for name, generator in all_generators.items():
        print(f"Processing dataset: {name}")
        dataset = generator()

        task_types = []
        if name in node_datasets:
            task_types.append('node')
        if name in edge_datasets:
            task_types.append('edge')
        if name in graph_datasets:
            task_types.append('graph')

        for task_type in task_types:
            split_data(dataset, name, task=task_type, train_ratio=0.2, K=5)
            split_dir = f"{ROOT_DIR}/datasets_split/{name}/split/"
            assert os.path.exists(split_dir), f"Split directory {split_dir} does not exist"
            split_files = [f for f in os.listdir(split_dir) if
                           f.endswith(".pt") and f.startswith(f"split_{task_type}_")]
            for split_file in split_files:
                split_path = os.path.join(split_dir, split_file)
                split_dict = torch.load(split_path)
                train_label2idx = split_dict['train_label2idx']
                for K in [1, 5]:
                    generate_few_shot_tasks(
                        train_label2idx=train_label2idx,
                        name=name,
                        task=task_type,
                        task_dim=int(split_file.split('_')[-1].split('.')[0]),
                        N=len(train_label2idx),  # choose all the classes
                        K=K,
                        num_tasks=50
                    )

