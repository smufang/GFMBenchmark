import torch
import numpy as np


class TemporalNeighborSampler:
    """
    Handles temporal graph sampling.
    Corresponds to 'NeighborFinder' in TF code.
    Used in DDGCL
    """
    def __init__(self, data):
        # Build adjacency list from the full dataset
        self.adj_list = {}
        
        # Convert to numpy for fast indexing on CPU
        src = data.src.numpy()
        dst = data.dst.numpy()
        t = data.t.numpy()
        msg = data.msg.numpy()
        
        print(">>> [Sampler] Building Adjacency Index...")
        for s, d, time, m in zip(src, dst, t, msg):
            if s not in self.adj_list: self.adj_list[s] = []
            if d not in self.adj_list: self.adj_list[d] = []
            self.adj_list[s].append((d, time, m))
            self.adj_list[d].append((s, time, m))
            
    def _sample_one_view(self, root_nodes, timestamps, n_neighbors=20):
        """Samples neighbors for a single view (Current or Past)."""
        edges = []
        edge_times = []
        edge_feats = []
        
        n_id = list(root_nodes)
        node_map = {n: i for i, n in enumerate(root_nodes)}
        
        for i, (node, t_cut) in enumerate(zip(root_nodes, timestamps)):
            node = int(node)
            if node not in self.adj_list: continue
            
            neighbors = self.adj_list[node]
            
            # Reverse traversal: find most recent past neighbors
            found = 0
            for neighbor, t_edge, feat in reversed(neighbors):
                if t_edge <= t_cut:
                    if neighbor not in node_map:
                        node_map[neighbor] = len(n_id)
                        n_id.append(neighbor)
                    
                    # Edge: Neighbor -> Root Node
                    edges.append([node_map[neighbor], i])
                    edge_times.append(t_cut - t_edge)
                    edge_feats.append(feat)
                    
                    found += 1
                    if found >= n_neighbors: break
        
        if not edges: return None
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_times = torch.tensor(edge_times, dtype=torch.float)
        edge_feats = torch.tensor(np.array(edge_feats), dtype=torch.float)
        n_id = torch.tensor(n_id, dtype=torch.long)
        
        return n_id, edge_index, edge_times, edge_feats

    def get_contrastive_batch(self, batch_src, batch_t, gamma=0.4):
        """
        Constructs two temporal views:
        1. Current View (at t)
        2. Past View (at cl_time)
        """
        batch_src_np = batch_src.cpu().numpy()
        batch_t_np = batch_t.cpu().numpy()
        
        # View 1: Current Time
        v1 = self._sample_one_view(batch_src_np, batch_t_np)
        
        dynamic_window = gamma * batch_t_np
        # View 2: Random Past Time (cl_time)
        rand_delta = np.random.uniform(0, dynamic_window)
        t_past_np = np.maximum(batch_t_np - rand_delta, 0)
        v2 = self._sample_one_view(batch_src_np, t_past_np)
        
        return v1, v2, torch.tensor(t_past_np, dtype=torch.float)