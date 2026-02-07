import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


class SubgraphBatchDataset(Dataset):
    def __init__(self, data, graph_label_index=None, graph_label=None, batch_size=1024,
                 shuffle=False, drop_last=False):
        """
        Dataset for sampling subgraphs from a large batched graph.

        This dataset is designed for scenarios where multiple small graphs have been
        merged into a single large graph using Batch.from_data_list(), and each node
        has a `batch` attribute indicating which original graph it belongs to.

        Features:
        - Samples graphs by `graph_label_index` with optional `graph_label`.
        - Supports shuffle, drop_last, and replacement (infinite sampling) modes.
        - Returns subgraphs as PyG Data objects with 'graph_label' and 'graph_label_index'.
        - If `graph_label` is None, `batch.graph_label` will be None.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if not hasattr(self.data, "batch"):
            raise ValueError("The input `data` must have a `batch` attribute.")

        if graph_label_index is None:
            self.graph_label_index = torch.unique(self.data.batch)
        else:
            self.graph_label_index = graph_label_index

        if graph_label is None:
            self.graph_label = None
        else:
            self.graph_label = graph_label
            assert len(self.graph_label_index) == len(self.graph_label), \
                f"graph_label_index and graph_label must have the same length, " \
                f"got {len(self.graph_label_index)} vs {len(self.graph_label)}"

        indices = torch.arange(len(self.graph_label_index))
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]

        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            end_idx = min(i + self.batch_size, len(indices))
            if self.drop_last and end_idx - i < self.batch_size:
                break
            self.batches.append(indices[i:end_idx])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.data
        batch_idx = self.batches[idx]
        batch_ids = self.graph_label_index[batch_idx]
        batch_labels = self.graph_label[batch_idx] if self.graph_label is not None else None

        mask = torch.isin(data.batch, batch_ids)
        node_idx = mask.nonzero(as_tuple=True)[0]
        if node_idx.numel() == 0:
            return None

        edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        batch = Data(
            x=data.x[node_idx],
            edge_index=edge_index,
            batch=data.batch[node_idx],
            graph_label=batch_labels,
            graph_label_index=batch_ids
        )
        return batch


def BatchGraphLoader(data, graph_label_index=None, graph_label=None,
                     batch_size=32, shuffle=False, drop_last=False, num_workers=0):
    dataset = SubgraphBatchDataset(
        data,
        graph_label_index=graph_label_index,
        graph_label=graph_label,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, persistent_workers=False)
    return loader
