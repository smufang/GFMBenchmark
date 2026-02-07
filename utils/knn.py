import torch
import torch.nn.functional as F


def knn_fast(x, k, block_size, eps=1e-8):
    """ 
    Build kNN graph using cosine similarity in a fast and memory-efficient way.
    Args:
        x: Node feature matrix of shape [num_nodes, num_features]
        k: Number of nearest neighbors
        block_size: Block size for processing nodes in chunks
    Returns:
        edge_index: Tensor of shape [2, num_edges] representing the kNN graph
        edge_weights: Tensor of shape [num_edges] representing edge weights
    """
    # L2 normalize
    x = F.normalize(x, dim=1, p=2)
    num_nodes = x.shape[0]
    device = x.device
    edges_per_node = min(k + 1, num_nodes-1)  # including self-loop
    total_edges = num_nodes * edges_per_node

    # Allocate buffers
    edge_weights = torch.zeros(total_edges, device=device)
    src_nodes = torch.zeros(total_edges, device=device)
    dst_nodes = torch.zeros(total_edges, device=device)

    # For degree-normalization
    degree_src = torch.zeros(num_nodes, device=device)
    degree_dst = torch.zeros(num_nodes, device=device)

    start = 0
    while start < num_nodes:
        end = min(start + block_size, num_nodes)

        # local block
        x_block = x[start:end]

        # cosine similarity
        similarities = x_block @ x.T

        # top-k neighbors (including itself)
        # top_vals: [block_size, edges_per_node]
        # top_inds: [block_size, edges_per_node]
        top_vals, top_inds = similarities.topk(k=edges_per_node, dim=-1)

        s = start * edges_per_node
        t = end * edges_per_node

        # flatten
        edge_weights[s:t] = top_vals.reshape(-1)
        src_nodes[s:t] = torch.arange(start, end, device=device).repeat_interleave(edges_per_node)
        dst_nodes[s:t] = top_inds.reshape(-1)

        # accumulate degrees
        degree_src[start:end] = torch.sum(top_vals, dim=1)
        degree_dst.index_add_(0, top_inds.reshape(-1), top_vals.reshape(-1))

        start += block_size

    # degree normalization
    degree = degree_src + degree_dst
    degree = degree.clamp(min=eps)
    src_nodes = src_nodes.long()
    dst_nodes = dst_nodes.long()

    # This normalization isn't used in here, as GCNConv() in PYG has its own normalization.
    edge_weights *= (torch.pow(degree[src_nodes], -0.5) * torch.pow(degree[dst_nodes], -0.5))

    # build edge_index
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    return edge_index, edge_weights
