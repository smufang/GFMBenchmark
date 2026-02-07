import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import random


class MultiGraphSampler:
    def __init__(self, data_len, batch_size, shuffle=True, replace=False, drop_last=False):
        self.data_len = data_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.drop_last = drop_last

    def __iter__(self):
        if self.replace:
            total_batches = self.data_len // self.batch_size if self.drop_last else (
                                                                                                self.data_len + self.batch_size - 1) // self.batch_size
            for _ in range(total_batches):
                yield [random.randint(0, self.data_len - 1) for _ in range(self.batch_size)]
        else:
            indices = list(range(self.data_len))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, self.data_len, self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        if self.replace or self.drop_last:
            return self.data_len // self.batch_size
        else:
            return (self.data_len + self.batch_size - 1) // self.batch_size


def identity_collate_fn(batch):
    return batch


def MultiGraphLoader(dataset, batch_size, shuffle=True, replace=False, drop_last=False, num_workers=0) -> DataLoader:
    sampler = MultiGraphSampler(len(dataset),
                                batch_size=batch_size,
                                shuffle=shuffle,
                                replace=replace,
                                drop_last=drop_last)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate_fn, num_workers=num_workers, persistent_workers=False)
    return loader


if __name__ == '__main__':
    def create_toy_graph(num_nodes, num_edges):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn((num_nodes, 4))
        return Data(x=x, edge_index=edge_index)


    dataset = [create_toy_graph(num_nodes=10 + i, num_edges=20 + i) for i in range(20)]
    sampler = MultiGraphSampler(len(dataset), batch_size=6, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate_fn)
    for i, graph_list in enumerate(loader):
        print(f"Batch {i + 1}:")
        for j, g in enumerate(graph_list):
            print(f"  Graph {j + 1}: num_nodes = {g.num_nodes}, num_edges = {g.num_edges}")
        print("=" * 40)
