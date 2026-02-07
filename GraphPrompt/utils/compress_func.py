import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy.sparse
from scipy.sparse.linalg import svds


def pad_to_size(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur_num, cur_dim = tensor.shape

    # Pad columns to the right
    if cur_dim < target_dim:
        pad_right = target_dim - cur_dim
        tensor = torch.cat([
            tensor,
            torch.zeros(cur_num, pad_right, device=tensor.device, dtype=tensor.dtype)
        ], dim=1)

    return tensor


def compress_pca(node_feature: torch.Tensor, k: int) -> torch.Tensor:
    """
    Align node features dimension by PCA. [num_nodes, input_dim] -> [num_nodes, k]
    """
    if node_feature.size(0) < k:
        raise ValueError(f"Number of nodes {node_feature.size(0)} is less than k {k}. PCA cannot be applied.")
    # CSR -> COO
    if node_feature.layout == torch.sparse_csr:
        node_feature = node_feature.to_sparse()

    # Sparse COO case
    if node_feature.layout == torch.sparse_coo:
        row = node_feature.indices()[0].numpy()
        col = node_feature.indices()[1].numpy()
        feature = node_feature.values().numpy()
        sp_mat = scipy.sparse.coo_matrix((feature, (row, col)), shape=node_feature.shape)
        # Note: We do NOT mean-center the sparse matrix here
        # because subtracting the mean would densify the matrix,
        # leading to high memory usage and low computational efficiency.
        svd = TruncatedSVD(n_components=k)
        x_reduced = svd.fit_transform(sp_mat)
        return torch.from_numpy(x_reduced).float()

    # Dense case
    else:
        node_feature = node_feature.float()
        x_pad = pad_to_size(node_feature, k).detach()
        pca = PCA(n_components=k)
        x_reduced = pca.fit_transform(x_pad.numpy())
        return torch.from_numpy(x_reduced).float()


def compress_svd(node_feature: torch.Tensor, k: int) -> torch.Tensor:
    """
    Align node features dimension by SVD. [num_nodes, input_dim] -> [num_nodes, k]
    """
    if node_feature.size(0) < k:
        raise ValueError(f"Number of nodes {node_feature.size(0)} is less than k {k}. SVD cannot be applied.")
    # CSR -> COO
    if node_feature.layout == torch.sparse_csr:
        node_feature = node_feature.to_sparse()

    # Sparse COO case
    if node_feature.layout == torch.sparse_coo:
        row = node_feature.indices()[0].numpy()
        col = node_feature.indices()[1].numpy()
        feature = node_feature.values().numpy()
        sp_mat = scipy.sparse.coo_matrix((feature, (row, col)), shape=node_feature.shape)
        U, S, _ = svds(sp_mat, k=k, which='LM')
        idx = np.argsort(-S)  # Sort S in descending order
        U, S = U[:, idx], S[idx]
        return torch.from_numpy(U @ np.diag(S)).float()

    # Dense case
    else:
        node_feature = node_feature.float()
        x_pad = pad_to_size(node_feature, k).detach()
        U, S, _ = torch.linalg.svd(x_pad, full_matrices=False)
        return U[:, :k] @ torch.diag(S[:k])
    
def gcope_svd(node_feature: torch.Tensor, k: int) -> torch.Tensor:
    if node_feature.size(-1) > k:
        node_feature = compress_svd(node_feature, k)
    elif node_feature.size(-1) < k:
        node_feature = pad_to_size(node_feature, k).float()
    else:
        node_feature = node_feature.float()
    return node_feature

if __name__ == '__main__':
    from data_provider.data_loader import get_original_data
    from data_provider import pretrain

    data_dict = get_original_data(pretrain)
    dataset = data_dict.values()
    for data in dataset:
        # data.to('cuda')
        print(data)
        data.x = compress_svd(data.x, 50)
        print(data)
