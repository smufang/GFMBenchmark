import torch
import torch.nn.functional as F

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        emb1: Tensor of shape [N, d]
        emb2: Tensor of shape [M, d]
    Returns:
        sim_matrix: Tensor of shape [N, M]
    """
    emb1_norm = F.normalize(emb1, dim=-1, eps=eps)
    emb2_norm = F.normalize(emb2, dim=-1, eps=eps)
    sim_matrix = torch.mm(emb1_norm, emb2_norm.t()) / temperature
    sim_matrix = torch.clamp(sim_matrix, min=-1.0 + eps, max=1.0 - eps)

    return sim_matrix 
