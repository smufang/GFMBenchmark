import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import random


class EdgeBasedGraphSampler:
    def __init__(self, dataset, max_edges, shuffle=True):
        """
        A custom sampler that groups graphs into batches
        based on the maximum number of edges allowed per batch.
        """
        self.dataset = dataset
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        batch, total_edges = [], 0
        i = 0
        while i < len(self.indices):
            idx = self.indices[i]
            graph = self.dataset[idx]
            num_edges = graph.num_edges

            # Case 1: If a single graph exceeds max_edges,
            # return it alone as a batch.
            if num_edges > self.max_edges:
                if batch:
                    yield batch
                    batch, total_edges = [], 0
                yield [idx]
                i += 1
                continue

            # Case 2: If adding this graph would exceed max_edges,
            # yield the current batch and DO NOT add this graph yet.
            if total_edges + num_edges > self.max_edges:
                if batch:
                    yield batch
                batch, total_edges = [], 0
                continue

            # Case 3: Safe to add this graph
            batch.append(idx)
            total_edges += num_edges
            i += 1

        # Yield any remaining graphs in the last batch
        if batch:
            yield batch

    def __len__(self):
        # Dynamic batching makes it hard to know in advance,
        # so we return the number of graphs as an approximation.
        return len(self.dataset)


def identity_collate_fn(batch):
    """Return the list of graphs as-is (no additional collation)."""
    return batch


def MultiGraphLoader(dataset, max_edges, shuffle=True, num_workers=0) -> DataLoader:
    """
    A DataLoader that batches graphs by maximum edge count instead of
    a fixed number of graphs.

    Args:
        dataset (list[Data]): List of PyG Data objects.
        max_edges (int): Maximum number of edges per batch.
        shuffle (bool): Shuffle graphs before batching.
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader with custom batching logic.
    """
    sampler = EdgeBasedGraphSampler(dataset, max_edges=max_edges, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
        persistent_workers=False
    )
    return loader


if __name__ == '__main__':
    def create_toy_graph(num_nodes, num_edges):
        """Utility function to create a toy graph with random edges and features."""
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn((num_nodes, 4))
        return Data(x=x, edge_index=edge_index)


    # Example dataset: a list of graphs with increasing edge counts
    dataset = [create_toy_graph(num_nodes=10 + i, num_edges=20 + i) for i in range(20)]

    # Create a loader that batches graphs based on max_edges
    loader = MultiGraphLoader(dataset, max_edges=80, shuffle=True)

    # Iterate through batches and print details
    for i, graph_list in enumerate(loader):
        total_edges = sum(g.num_edges for g in graph_list)
        print(f"Batch {i}: {len(graph_list)} graphs, {total_edges} edges")
