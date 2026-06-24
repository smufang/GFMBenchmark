import random

from torch.utils.data import DataLoader


class SizeBasedGraphSampler:
    def __init__(self, dataset, max_edges, max_nodes, shuffle=True):
        """
        A custom sampler that groups graphs into batches based on both maximum
        edge count and maximum node count.
        """
        self.dataset = dataset
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def _graph_size(self, graph):
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        if num_nodes is None:
            raise ValueError("Graph num_nodes is required for SizeBasedGraphSampler.")
        return num_nodes, num_edges

    def _single_exceeds_limit(self, num_nodes, num_edges):
        return num_nodes > self.max_nodes or num_edges > self.max_edges

    def _batch_exceeds_limit(self, total_nodes, total_edges, num_nodes, num_edges):
        return (
            total_nodes + num_nodes > self.max_nodes
            or total_edges + num_edges > self.max_edges
        )

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        batch, total_nodes, total_edges = [], 0, 0
        i = 0
        while i < len(self.indices):
            idx = self.indices[i]
            graph = self.dataset[idx]
            num_nodes, num_edges = self._graph_size(graph)

            # This sampler can group graphs, but cannot split one oversized graph.
            if self._single_exceeds_limit(num_nodes, num_edges):
                if batch:
                    yield batch
                    batch, total_nodes, total_edges = [], 0, 0
                yield [idx]
                i += 1
                continue

            if self._batch_exceeds_limit(total_nodes, total_edges, num_nodes, num_edges):
                if batch:
                    yield batch
                batch, total_nodes, total_edges = [], 0, 0
                continue

            batch.append(idx)
            total_nodes += num_nodes
            total_edges += num_edges
            i += 1

        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset)


def identity_collate_fn(batch):
    """Return the list of graphs as-is."""
    return batch


def MultiGraphLoader(dataset, max_edges, max_nodes, shuffle=True, num_workers=0):
    """
    A DataLoader that batches graphs by maximum edge and node counts.
    """
    sampler = SizeBasedGraphSampler(
        dataset,
        max_edges=max_edges,
        max_nodes=max_nodes,
        shuffle=shuffle,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
    )
    return loader
