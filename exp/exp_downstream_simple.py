from exp.exp_basic import ExpBasic
import torch
from torch_geometric.utils import subgraph, is_undirected
from data_provider.subbatch_loader import BatchGraphLoader
from torch_geometric.data import Data, Batch
import os
import time
from tqdm import tqdm


class ExpDownstreamBatch(ExpBasic):
    def __init__(self, args, name, dataset, pretrain_dict=None):
        args.target_data = name
        super(ExpDownstreamBatch, self).__init__(args, pretrain_dict)
        self._print_main(f"Target dataset: {name}")

        self.need_y = False

        self.dataset = dataset()
        preprocess_map = {
            "basic": self._basic_preprocess,
            "simple": self._simple_preprocess,
        }
        try:
            self.data = preprocess_map[args.preprocess](self.dataset)
        except KeyError:
            raise ValueError(f"Unknown preprocess mode: {args.preprocess}")

        if self.compress_func is not None:
            self.data = self._pure_data(self._get_compressed_data(self.data))
        else:
            self.data.x = self.data.x.float()
        
        if self.args.model == "gcope" and self.args.task_name != "graph":
            self.data_lst = self._read_induced_graphs()

        if args.preprocess == 'basic':
            self.avg_degree = self.data.num_edges / self.data.num_nodes
            self._print_main(f"Average degree: {self.avg_degree:.2f}")
            self._write_log(f"Average degree: {self.avg_degree:.2f}")
            self.is_undirected = is_undirected(self.data.edge_index)
            self._print_main(f"Is undirected: {self.is_undirected}")
            self._write_log(f"Is undirected: {self.is_undirected}")

    def _get_few_shot(self):
        from data_provider.fewshot_generator import load_few_shot_tasks
        return load_few_shot_tasks(
            self.args.target_data, task=self.args.task_name, K=self.args.num_shots, num_tasks=self.args.num_tasks
        )

    def _get_test(self):
        from data_provider.fewshot_generator import load_test_splits
        return load_test_splits(self.args.target_data, task=self.args.task_name)

    def _create_progress_bar(self, few_shot_tasks, task_dim):
        if self._is_main_process():
            return tqdm(
                few_shot_tasks,
                desc=f"Task Dim {task_dim}",
                position=0,
                leave=True,
                dynamic_ncols=False,
            )
        else:
            return few_shot_tasks

    def _get_loader_subgraph(self, data, task_labels, task_ids, is_train):
        # batch: keep the original batch index
        batch_size = self.args.batch_size
        persistent_workers=False
        if self.args.task_name == "node":
            loader = BatchGraphLoader(
                Batch.from_data_list(self.data_lst),
                graph_label_index=task_ids,
                graph_label=task_labels,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
            for batch in loader:
                yield batch, batch.graph_label, batch.graph_label_index

        elif self.args.task_name == "edge":
            data_batch = Batch.from_data_list(self.data_lst)
            for i in range(0, len(task_ids), batch_size):
                current_bs = min(batch_size, len(task_ids) - i)
                batch_ids = task_ids[i:i + current_bs]
                batch_labels = task_labels[i:i + current_bs]

                graph_edge_index = data.edge_index[:, batch_ids]
                src, dst = graph_edge_index
                involved_nodes = torch.unique(torch.cat([src, dst]))

                mask = torch.isin(data_batch.batch, involved_nodes)
                node_idx = mask.nonzero(as_tuple=True)[0]
                edge_index, _ = subgraph(node_idx, data_batch.edge_index, relabel_nodes=True)

                batch = Data(
                    x=data_batch.x[node_idx],
                    edge_index=edge_index,
                    batch=data_batch.batch[node_idx]
                )

                yield batch, batch_labels, graph_edge_index

        elif self.args.task_name == "graph":
            loader = BatchGraphLoader(
                data,
                graph_label_index=task_ids,
                graph_label=task_labels,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
            for batch in loader:
                yield batch, batch.graph_label, batch.graph_label_index

    def _induced_graphs(self, data, smallest_size=10, largest_size=30):

        from torch_geometric.utils import subgraph, k_hop_subgraph
        from torch_geometric.data import Data
        import numpy as np

        induced_graph_list = []
        total_node_num = data.num_nodes
        if self.data_type == "hetero":
            # All the main node type for 'hetero' is placed at the first position
            target = data.raw_texts[0]
            total_node_num = data.raw_texts.count(target)
            
        time_start = time.time()
        current_time = time.time()
        #total_node_idx = (data.y != -1).nonzero(as_tuple=True)[0]
        for index in range(total_node_num):
            current_hop = 2
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index, relabel_nodes=True)
            
            while len(subset) < smallest_size and current_hop < 5 and len(subset) > 1:
                current_hop += 1
                subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                    edge_index=data.edge_index)

            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

            x = data.x[subset]

            induced_graph = Data(x=x, edge_index=sub_edge_index)
            induced_graph_list.append(induced_graph)
            if(index%1000==0):
                self._print_main(f'Generated {index}/{total_node_num} subgraph data, Time: {time.time() - current_time:.2f}s')
                current_time = time.time()

        self._print_main(f'Generated {index}/{total_node_num} subgraph data, Time: {time.time() - time_start:.2f}s')
        self._write_log(f'Generated {index}/{total_node_num} subgraph data, Time: {time.time() - time_start:.2f}s')
        save_dir = os.path.join(os.getcwd(), 'datasets/{}/preprocess'.format(self.args.target_data))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'induced_graphs.pt')
        torch.save(induced_graph_list, save_path)
        self._print_main('Saved induced graphs to {}'.format(save_path))
    
    def _read_induced_graphs(self):
        induced_path = os.path.join(os.getcwd(), 'datasets/{}/preprocess/induced_graphs.pt'.format(self.args.target_data))
        if not os.path.exists(induced_path):
            self._induced_graphs(self.data)
        return torch.load(induced_path)
