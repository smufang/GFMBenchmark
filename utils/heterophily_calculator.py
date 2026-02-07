from torch_geometric.data import HeteroData, TemporalData, Data
from torch_geometric.utils import homophily


def homophily_data(data, method):
    if not hasattr(data, 'y') or data.y is None:
        # no label in data
        return -1

    if data.y.shape[0] == 1:
        # graph label
        return -2

    return homophily(data.edge_index, data.y, method=method)


def edge_homophily_hetero(data):
    tot_diff = 0
    tot_edges = 0
    for etype in data.edge_types:
        src, rel, dst = etype
        edge_index = data[etype].edge_index
        y_src = data[src].y if hasattr(data[src], 'y') else None
        y_dst = data[dst].y if hasattr(data[dst], 'y') else None

        if y_src is None or y_dst is None:
            continue

        diff = (y_src[edge_index[0]] == y_dst[edge_index[1]]).sum().item()
        tot = edge_index.size(1)
        tot_diff += diff
        tot_edges += tot

    if tot_edges > 0:
        return tot_diff / tot_edges
    else:
        return -1


def node_homophily_hetero(data):
    total = 0
    count = 0
    for ntype in data.node_types:
        if not hasattr(data[ntype], 'y'):
            continue
        num_nodes = data[ntype].num_nodes
        y_dst = data[ntype].y
        same = torch.zeros(num_nodes)
        deg = torch.zeros(num_nodes)

        for src, rel, dst in data.edge_types:
            if dst != ntype or src != ntype:
                continue

            edge_index = data[(src, rel, dst)].edge_index
            y_src = data[src].y
            label_match = (y_src[edge_index[0]] == y_dst[edge_index[1]])
            same.index_add_(0, edge_index[1], label_match.float())
            deg.index_add_(0, edge_index[1], torch.ones_like(label_match, dtype=torch.float))

        ratio = same / deg.clamp(min=1)
        total += ratio.sum().item()
        count += num_nodes

    if count > 0:
        return total / count
    else:
        return -1


def compute_homophily(dataset):
    edge_total = 0
    node_total = 0
    n = len(dataset)
    for data in dataset:
        if isinstance(data, HeteroData):
            edge_ratio = edge_homophily_hetero(data)
        elif isinstance(data, Data):
            edge_ratio = homophily_data(data, 'edge')
        elif isinstance(data, TemporalData):
            edge_ratio = -3
        else:
            edge_ratio = -4

        if isinstance(data, HeteroData):
            node_ratio = node_homophily_hetero(data)
        elif isinstance(data, Data):
            node_ratio = homophily_data(data, 'node')
        elif isinstance(data, TemporalData):
            node_ratio = -3
        else:
            node_ratio = -4

        edge_total += edge_ratio
        node_total += node_ratio

    return edge_total / n, node_total / n


if __name__ == '__main__':
    from data_provider.data_generator import *

    datasets = {
    'Actor': actor,
    'Texas': texas,
    'Amazon-HeTGB': amazonh,
    'ogbn_arxiv': ogbn_arxiv
}

    print('-1 means no label\n'
          '-2 means only graph label\n'
          '-3 means temporal data\n'
          '-4 means other graph format')
    for name, generator in datasets.items():
        dataset = generator()
        edge, node = compute_homophily(dataset)
        print(f'The homophily ratio of {name} is {edge:.4f} based on edge and {node:.4f} based on node.')
