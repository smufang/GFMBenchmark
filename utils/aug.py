import torch
import random
import numpy as np
from torch_geometric.data import Data


class DataAugment:
    def __init__(self, data: Data, drop_percent=0.1):
        self.drop_percent = drop_percent
        self.node_feature = data.x  # [num_nodes, feat_dim]
        self.edge_index = data.edge_index  # [2, num_edges]
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.name = data.name  # [num_nodes]
        self.unique_names = list(set(self.name))

    def aug_random_mask(self):
        aug_feature = self.node_feature.clone()
        num_mask = int(self.num_nodes * self.node_feature)
        mask_ids = random.sample(range(self.num_nodes), num_mask)
        aug_feature[mask_ids, :] = 0.0
        return aug_feature

    # def aug_random_edge(self):
    #     edge_list = self.edge_index.t().tolist()
    #     num_add_drop = int(self.num_edges * self.drop_percent)

    #     # Remove edges (view all as directed graph)
    #     drop_ids = set(random.sample(range(self.num_edges), num_add_drop))
    #     keep_ids = [i for i in range(self.num_edges) if i not in drop_ids]
    #     kept_edges = [edge_list[i] for i in keep_ids]

    #     # Add edges
    #     orig_edges = set(map(tuple, edge_list))
    #     add_edges_set = set()
    #     attempts = 0
    #     max_attempts = num_add_drop * 10

    #     while len(add_edges_set) < num_add_drop and attempts < max_attempts:
    #         u = random.randint(0, self.num_nodes - 1)
    #         v = random.randint(0, self.num_nodes - 1)
    #         if u != v and (u, v) not in orig_edges and (u, v) not in add_edges_set:
    #             add_edges_set.add((u, v))
    #         attempts += 1

    #     add_edges = list(add_edges_set)
    #     final_edges = kept_edges + add_edges
    #     aug_edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()

    #     return aug_edge_index
    def aug_random_edge(self):
        edge_list = self.edge_index.t().tolist()
        name = np.array(self.name)
        final_edges = []

        for n in self.unique_names:
            node_idx = np.where(name == n)[0]
            node_idx_set = set(node_idx)
            sub_edges = [e for e in edge_list if e[0] in node_idx_set and e[1] in node_idx_set]
            num_edges = len(sub_edges)
            num_add_drop = int(num_edges * self.drop_percent)

            # Remove edges
            drop_ids = set(random.sample(range(num_edges), num_add_drop))
            keep_ids = [i for i in range(num_edges) if i not in drop_ids]
            kept_edges = [sub_edges[i] for i in keep_ids]

            # Add edges
            orig_edges = set(map(tuple, sub_edges))
            add_edges_set = set()
            attempts = 0
            max_attempts = num_add_drop * 10

            while len(add_edges_set) < num_add_drop and attempts < max_attempts:
                u = random.choice(node_idx)
                v = random.choice(node_idx)
                if u != v and (u, v) not in orig_edges and (u, v) not in add_edges_set:
                    add_edges_set.add((u, v))
                attempts += 1

            add_edges = list(add_edges_set)
            final_edges.extend(kept_edges + add_edges)

        aug_edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()
        return aug_edge_index

    def _node_texts(self, keep_ids, num_add=0):
        aug_name = [self.name[i] for i in sorted(keep_ids)]
        if num_add > 0:
            add_name = random.choices(self.unique_names, k=num_add)
            aug_name = aug_name + add_name
        return aug_name

    @staticmethod
    def _delete_row_col(input_matrix, drop_list):
        remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
        out = input_matrix[remain_list, :]
        return out

    def aug_drop_node(self):
        num_drop = int(self.num_nodes * self.drop_percent)
        drop_ids = set(random.sample(range(self.num_nodes), num_drop))
        keep_ids = [i for i in range(self.num_nodes) if i not in drop_ids]

        aug_feature = self._delete_row_col(self.node_feature, drop_ids)

        # Filter edges that do not contain any dropped nodes
        mask = [
            (u not in drop_ids) and (v not in drop_ids)
            for u, v in self.edge_index.t().tolist()
        ]
        kept_edges = self.edge_index[:, mask]

        # Remap node IDs to ensure consecutive numbering
        id_map = {old: new for new, old in enumerate(keep_ids)}
        aug_edge_index = torch.tensor(
            [[id_map[u], id_map[v]] for u, v in kept_edges.t().tolist()],
            dtype=torch.long
        ).t().contiguous()

        aug_name = self._node_texts(keep_ids)

        return aug_feature, aug_edge_index, aug_name

    def aug_subgraph(self):
        # Number of nodes in subgraph
        target_num_nodes = int(self.num_nodes * (1 - self.drop_percent))

        # Pick a random center node
        center_node_id = random.randint(0, self.num_nodes - 1)
        sub_node_ids = [center_node_id]
        visited = set(sub_node_ids)

        # BFS-like expansion
        edge_list = self.edge_index.t().tolist()
        neighbors_dict = {}
        for u, v in edge_list:
            neighbors_dict.setdefault(u, []).append(v)
            neighbors_dict.setdefault(v, []).append(u)  # if undirected

        frontier = [center_node_id]
        while len(sub_node_ids) < target_num_nodes and frontier:
            new_frontier = []
            for node in frontier:
                neighbors = neighbors_dict.get(node, [])
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        sub_node_ids.append(n)
                        new_frontier.append(n)
                        if len(sub_node_ids) >= target_num_nodes:
                            break
                if len(sub_node_ids) >= target_num_nodes:
                    break
            frontier = new_frontier

        # Remove dropped nodes from features
        drop_node_ids = sorted(set(range(self.num_nodes)) - set(sub_node_ids))
        aug_feature = self._delete_row_col(self.node_feature, drop_node_ids)

        # Filter edges inside subgraph
        id_map = {old: new for new, old in enumerate(sub_node_ids)}
        mask = [(u in id_map and v in id_map) for u, v in edge_list]
        kept_edges = [(id_map[u], id_map[v]) for (u, v), m in zip(edge_list, mask) if m]
        aug_edge_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()

        aug_name = self._node_texts(sub_node_ids)

        return aug_feature, aug_edge_index, aug_name

    def build_aug(self, aug_type):
        # idx = np.random.permutation(self.num_nodes)
        # shuffled_features = self.node_feature[idx, :]

        shuffled_features = torch.zeros_like(self.node_feature)
        name = np.array(self.name)
        for n in self.unique_names:
            idx = np.where(name == n)[0]
            shuffled_idx = np.random.permutation(idx)
            shuffled_features[idx] = self.node_feature[shuffled_idx]


        lbl_1 = torch.ones(self.num_nodes)
        lbl_2 = torch.zeros(self.num_nodes)
        lbl = torch.cat((lbl_1, lbl_2), dim=0)

        if aug_type == 'edge':
            aug_edge_index1 = self.aug_random_edge()
            aug_edge_index2 = self.aug_random_edge()
            return Data(shuffled_x=shuffled_features,
                        edge_index_aug1=aug_edge_index1, edge_index_aug2=aug_edge_index2,
                        lbl=lbl)

        elif aug_type == 'mask':
            aug_feature1 = self.aug_random_mask()
            aug_feature2 = self.aug_random_mask()
            return Data(shuffled_x=shuffled_features, aug_feature1=aug_feature1, aug_feature2=aug_feature2,
                        lbl=lbl)

        elif aug_type == 'node':
            aug_feature1, aug_edge_index1, aug_name1 = self.aug_drop_node()
            aug_feature2, aug_edge_index2, aug_name2 = self.aug_drop_node()
            return Data(shuffled_x=shuffled_features, aug_feature1=aug_feature1, aug_feature2=aug_feature2,
                        edge_index_aug1=aug_edge_index1, edge_index_aug2=aug_edge_index2,
                        name_aug1=aug_name1, name_aug2=aug_name2,
                        lbl=lbl)

        elif aug_type == 'subgraph':
            aug_feature1, aug_edge_index1, aug_name1 = self.aug_subgraph()
            aug_feature2, aug_edge_index2, aug_name2 = self.aug_subgraph()
            return Data(shuffled_x=shuffled_features, aug_feature1=aug_feature1, aug_feature2=aug_feature2,
                        edge_index_aug1=aug_edge_index1, edge_index_aug2=aug_edge_index2,
                        name_aug1=aug_name1, name_aug2=aug_name2,
                        lbl=lbl)

        else:
            raise ValueError(f"Unsupported augmentation type: {aug_type}")


if __name__ == '__main__':
    from data_provider.data_generator import *

    dataset = {'ACM': acm,
               'DBLP': dblp,
               'BZR': bzr,
               'COX2': cox2}
    from data_provider.data_loader_d import pretrain_loader

    dataloader = pretrain_loader(dataset, batch_size=2)
    hidden_dim = 64
    with torch.no_grad():
        for data in dataloader:
            augmentor = DataAugment(data, drop_percent=0.2)
            aug_edge_index = augmentor.aug_random_edge()
            orig_edges = set(map(tuple, data.edge_index.t().tolist()))
            aug_edges = set(map(tuple, aug_edge_index.t().tolist()))
            changed_edges = orig_edges.symmetric_difference(aug_edges)

            print(f"原始边数: {len(orig_edges)}")
            print(f"增强后边数: {len(aug_edges)}")
            print(f"改动的边数: {len(changed_edges)}")
            print(len(changed_edges) / len(orig_edges))
            print('-------------------')
