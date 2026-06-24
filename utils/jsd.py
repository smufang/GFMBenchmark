import csv
import json
import os
from itertools import combinations
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, HeteroData, TemporalData
from torch_geometric.utils import degree

from utils.format_trans import hetero_to_data, multi_to_one, temporal_to_data

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _in_degree_distribution(edge_index: torch.Tensor, num_nodes: int, max_degree: int) -> torch.Tensor:
    """Calculate the normalized in-degree probability distribution for a single graph."""
    # edge_index[1] represents target nodes (in-degree)
    deg = degree(edge_index[1], num_nodes=num_nodes).to(torch.long)
    
    # Cap degrees at max_degree to align all distributions
    deg = torch.clamp(deg, min=0, max=max_degree)
    
    # Count frequencies of each degree
    hist = torch.bincount(deg, minlength=max_degree + 1).float()
    
    # Normalize to create a probability distribution, avoiding division by zero
    return hist / hist.sum().clamp_min(1e-12)


def _jsd_from_distributions(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon Divergence between two probability distributions."""
    # Clamp to avoid log(0) errors
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    
    # Calculate the midpoint distribution
    m = 0.5 * (p + q)
    
    # Compute KL divergence for both sides
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
    
    return float(0.5 * (kl_pm + kl_qm))


def _extract_edge_index_num_nodes(dataset) -> Tuple[torch.Tensor, int]:
    if len(dataset) > 1:
        data = multi_to_one(dataset, need_y=False)
    else:
        data = dataset[0]
        if isinstance(data, TemporalData):
            data = temporal_to_data(data, need_y=False)
        elif isinstance(data, HeteroData):
            data = hetero_to_data(data, need_y=False)
        elif not isinstance(data, Data):
            raise ValueError(f"Unsupported data type: {type(data)}")

    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if edge_index is None:
        raise ValueError("edge_index is missing in processed dataset")
    if num_nodes is None:
        if edge_index.numel() == 0:
            raise ValueError("num_nodes is missing and edge_index is empty")
        num_nodes = int(edge_index.max().item()) + 1

    return edge_index, int(num_nodes)


def compute_all_in_degree_distributions(generators) -> Dict[str, torch.Tensor]:
    """Extract data, find global max degree, and compute aligned distributions."""
    # Step 1: Find the global max degree across ALL datasets to align bins
    max_degree = 0
    extracted: Dict[str, Tuple[torch.Tensor, int]] = {}
    items = list(generators.items())
    if tqdm:
        pbar = tqdm(items, desc="Load datasets", unit="dataset")
        for name, generator in pbar:
            pbar.set_postfix_str(f"current={name}")
            edge_index, num_nodes = _extract_edge_index_num_nodes(generator())
            extracted[name] = (edge_index, num_nodes)
            if edge_index.numel() > 0:
                dmax = int(degree(edge_index[1], num_nodes=num_nodes).max().item())
                max_degree = max(max_degree, dmax)
    else:
        for name, generator in items:
            print(f"[Load datasets] current={name}")
            edge_index, num_nodes = _extract_edge_index_num_nodes(generator())
            extracted[name] = (edge_index, num_nodes)
            if edge_index.numel() > 0:
                dmax = int(degree(edge_index[1], num_nodes=num_nodes).max().item())
                max_degree = max(max_degree, dmax)

    # Step 2: Compute degree distributions using the shared max_degree
    dist_dict: Dict[str, torch.Tensor] = {}
    items2 = list(extracted.items())
    if tqdm:
        pbar2 = tqdm(items2, desc="Build distributions", unit="dataset")
        for name, (edge_index, num_nodes) in pbar2:
            pbar2.set_postfix_str(f"current={name}")
            dist_dict[name] = _in_degree_distribution(edge_index, num_nodes, max_degree=max_degree)
    else:
        for name, (edge_index, num_nodes) in items2:
            print(f"[Build distributions] current={name}")
            dist_dict[name] = _in_degree_distribution(edge_index, num_nodes, max_degree=max_degree)
        
    return dist_dict


def save_degree_distributions_json(dist_dict: Dict[str, torch.Tensor], output_json: str) -> None:
    """Save the aligned degree distributions to a JSON file."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    payload = {
        "distributions": [
            {
                "name": name,
                "degree_distribution": dist.tolist(),
            }
            for name, dist in sorted(dist_dict.items())
        ]
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsd_matrix_csv(dist_dict: Dict[str, torch.Tensor], output_csv: str) -> None:
    """Compute pairwise JSD and save as a symmetric CSV matrix."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    names: List[str] = sorted(dist_dict.keys())
    n = len(names)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    # Compute upper triangle and mirror to lower triangle
    pairs = list(combinations(range(n), 2))
    if tqdm:
        pbar3 = tqdm(pairs, desc="Compute JSD matrix", unit="pair")
        for i, j in pbar3:
            pbar3.set_postfix_str(f"{names[i]} vs {names[j]}")
            v = _jsd_from_distributions(dist_dict[names[i]], dist_dict[names[j]])
            matrix[i][j] = v
            matrix[j][i] = v
    else:
        for i, j in pairs:
            v = _jsd_from_distributions(dist_dict[names[i]], dist_dict[names[j]])
            matrix[i][j] = v
            matrix[j][i] = v

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + names)
        for i, name in enumerate(names):
            writer.writerow([name] + matrix[i])


def run_structure_jsd_pipeline(
    generators: Dict,
    degree_json: str = "z_temp/in_degree_distribution.json",
    jsd_matrix_csv: str = "z_temp/structure_jsd_matrix.csv",
) -> Tuple[str, str]:
    """Execute the full pipeline: compute, save JSON, and save CSV."""
    dist_dict = compute_all_in_degree_distributions(generators)
    save_degree_distributions_json(dist_dict, degree_json)
    save_jsd_matrix_csv(dist_dict, jsd_matrix_csv)
    return degree_json, jsd_matrix_csv

if __name__ == "__main__":
    from data_provider import all_datasets
    run_structure_jsd_pipeline(all_datasets)
    print("JSD pipeline completed. Results saved to z_temp/")