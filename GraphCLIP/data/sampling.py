import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, remove_isolated_nodes, dropout_adj, remove_self_loops, k_hop_subgraph, to_edge_index, to_dgl
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
from torch_sparse import SparseTensor
from tqdm import tqdm


def ego_graphs_sampler_relabel(node_idx, data, hop=2, sparse=False, transform=None):
    ego_graphs = []
    edge_index = data.edge_index
    for idx in tqdm(node_idx.numpy().tolist(), desc="Generating subgraphs", total=node_idx.shape[0]):
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=True)
        g = Data(x=data.x[subset], edge_index=sub_edge_index, root_n_index=mapping, original_idx=subset)
        if transform is not None:
            g = transform(g) # e.g., add PE feature
        ego_graphs.append(g)
    return ego_graphs


def pyg_random_walk_relabel(seeds, data, length=5, restart_prob=0.8, is_undirected=True, transform=None):
    """
    For each seed node, perform random walk sampling of subgraphs and return Data(x, edge_index, root_n_index).
    """
    edge_index = data.edge_index
    node_num = data.num_nodes
    start_nodes = seeds
    graph_num = start_nodes.shape[0]

    # build SparseTensor
    value = torch.arange(edge_index.size(1))
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                             value=value, sparse_sizes=(node_num, node_num)).t()

    # 预先计算每个节点的入度
    src, dst = edge_index
    in_degree = torch.zeros(node_num, dtype=torch.long)
    in_degree.scatter_add_(0, dst, torch.ones_like(dst))

    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)

    # random walk
    for i in range(length):
        seed = torch.rand([graph_num])
        # 检查当前节点是否有入边
        has_neighbors = in_degree[current_nodes] > 0
        nei = adj_t.sample(1, current_nodes).squeeze()
        # 对于没有入边的节点，强制 restart
        sign = (seed < restart_prob) | (~has_neighbors)
        nei[sign] = start_nodes[sign]

        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei

    history = history.T
    signs = signs.T

    ego_graphs = []
    for i in tqdm(range(graph_num), desc="Generating subgraphs", total=graph_num):
        path = history[i]
        sign = signs[i]
        nodes = path.unique()

        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        # sub_edges = torch.LongTensor([sources, targets])
        sub_edges = torch.IntTensor([targets, sources]).long() # transpose for correct direction as we use adj_t above
        sub_edges = sub_edges.T[~sign[1:]].T  # remove edges where restart occurred
        if sub_edges.numel() == 0:
            seed = path[0].item()
            sub_edges = torch.tensor([[seed],[seed]], dtype=torch.long)

        # filter edges within the subgraph nodes
        mask = torch.isin(sub_edges[0], nodes) & torch.isin(sub_edges[1], nodes)
        sub_edges = sub_edges[:, mask]

        # relabel
        node_map = {old.item(): i for i, old in enumerate(nodes)}
        sub_edges = torch.stack([
            torch.tensor([node_map[int(u)] for u in sub_edges[0]]),
            torch.tensor([node_map[int(v)] for v in sub_edges[1]])
        ], dim=0)

        if is_undirected:
            sub_edges = to_undirected(sub_edges)

        sub_x = data.x[nodes]
        root_idx = torch.tensor([node_map[int(start_nodes[i])]])
        g = Data(x=sub_x, edge_index=sub_edges, root_n_index=root_idx)
        if transform is not None:
            g = transform(g) # e.g., add PE feature
        ego_graphs.append(g)

    return ego_graphs


def pyg_random_walk(seeds, graph, length, restart_prob=0.8, is_undirected=True):
    # 构建random walk子图
    edge_index = graph.edge_index
    node_num = graph.num_nodes
    start_nodes = seeds
    graph_num = start_nodes.shape[0]

    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t() # change: add .t() as not suitable for directed graph to find in-neighbors
    
    # 预先计算每个节点的入度
    src, dst = edge_index
    in_degree = torch.zeros(node_num, dtype=torch.long)
    in_degree.scatter_add_(0, dst, torch.ones_like(dst))
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    
    for i in range(length):
        seed = torch.rand([graph_num])
        # 检查当前节点是否有入边
        has_neighbors = in_degree[current_nodes] > 0
        nei = adj_t.sample(1, current_nodes).squeeze()
        # 对于没有入边的节点，强制 restart
        sign = (seed < restart_prob) | (~has_neighbors)
        nei[sign] = start_nodes[sign]

        # change as the old one has problem when some nodes have no in-neighbors (directed graph)
        # sign = seed < restart_prob
        # nei[sign] = start_nodes[sign]
        # Sample one neighbor for each node in current_nodes
        # Note: SparseTensor.sample never returns empty, even if a node has no outgoing edges.
        #       In that case, it returns a "dummy neighbor" or fills with a pseudo-node.

        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei

    history = history.T
    signs = signs.T

    node_list = []
    edge_list = []
    for i in tqdm(range(graph_num), desc="Generating subgraphs", total=graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        node_list.append(node_idx)

        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        # sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = torch.IntTensor([targets, sources]).long() # change: transpose for correct direction as we use adj_t above for directed graph
        sub_edges = sub_edges.T[~sign[1:]].T # 移除 restart 的边
        if sub_edges.numel() == 0:
            seed = path[0].item()
            sub_edges = torch.tensor([[seed],[seed]], dtype=torch.long)
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = torch.unique(sub_edges, dim=1)
            if is_undirected:
                sub_edges = to_undirected(sub_edges)
        edge_list.append(sub_edges)
    return node_list, edge_list


def RWR_sampler(selected_ids, graph, walk_steps=256, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_ids # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
        
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T

    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        # place the targe index in the first place
        target_idx = path[0].item()
        pos = torch.where(node_idx==target_idx)[0].item()
        if pos != 0:
            tmp = node_idx[0].item()
            node_idx[0] = target_idx
            node_idx[pos] = tmp
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())
        view['center_idx'] = target_idx
        view['neig_idx'] = node_idx
        # variables with 'index' will be automatically increased in data loader
        # view = Data(edge_index=sub_edges, x=graph.x[node_idx], center_index=target_idx, center_idx=target_idx, neig_idx=node_idx, y=graph.y[target_idx])

        graph_list.append(view)
    return graph_list

def add_remaining_selfloop_for_isolated_nodes(edge_index, num_nodes):
    num_nodes = max(maybe_num_nodes(edge_index), num_nodes)
    # only add self-loop on isolated nodes
    # edge_index, _ = remove_self_loops(edge_index)
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    connected_nodes_indices = torch.cat([edge_index[0], edge_index[1]]).unique()
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[connected_nodes_indices] = False
    loops_for_isolatd_nodes = loop_index[mask]
    loops_for_isolatd_nodes = loops_for_isolatd_nodes.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops_for_isolatd_nodes], dim=1)
    return edge_index

    
    
def collect_subgraphs(selected_id, graph, walk_steps=20, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_id # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T
    
    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())

        graph_list.append(view)
    return graph_list
        
def adjust_idx(edge_index, node_idx, full_g, center_idx):
    '''re-index the nodes and edge index

    In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
    nodes' index to edge index
    '''
    # # put center node in the first place
    # pos = torch.where(node_idx==center_idx)[0].item()
    # if pos != 0:
    #     tmp = node_idx[0]
    #     node_idx[0] = center_idx
    #     node_idx[pos] = tmp
    node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
    sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
    target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))
    edge_index = torch.IntTensor([sources_idx, target_idx]).long()
    x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
    return x_view

def ego_graphs_sampler(node_idx, data, hop=2, sparse=False):
    ego_graphs = []
    if sparse:
        edge_index, _ = to_edge_index(data.edge_index)
    else:
        edge_index  = data.edge_index
    for idx in tqdm(node_idx.numpy().tolist(), desc="Generating subgraphs", total=node_idx.shape[0]):
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=False) # no relabel
        # sub_edge_index = to_undirected(sub_edge_index)
        sub_x = data.x[subset]
        # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
        g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, original_idx=subset) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
        g['center_idx'] = idx
        g['neig_idx'] = subset
        ego_graphs.append(g)
    return ego_graphs


# def ego_graphs_sampler(node_idx, data, hop=2, sparse=False):
#     ego_graphs = []
#     if sparse:
#         edge_index, _ = to_edge_index(data.edge_index)
#     else:
#         edge_index  = data.edge_index
#     row, col = edge_index
#     num_nodes = data.x.shape[0]
#     for idx in node_idx.numpy().tolist():
#         subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=False)
#         # sub_edge_index = to_undirected(sub_edge_index)
#         pos = torch.where(idx==subset)[0].item()
#         if pos != 0:
#             tmp = subset[0].item()
#             subset[0] = idx
#             subset[pos] = tmp
#         sub_x = data.x[subset]
#         mapping = row.new_full((num_nodes, ), -1)
#         mapping[subset] = torch.arange(subset.size(0), device=row.device)
#         sub_edge_index = mapping[sub_edge_index]

#         # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
#         g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=subset) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
#         g['center_idx'] = idx
#         g['neig_idx'] = subset
#         ego_graphs.append(g)
#     return ego_graphs