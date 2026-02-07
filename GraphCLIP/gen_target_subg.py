import os
from multiprocessing import Pool
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from GraphCLIP.data.sampling import ego_graphs_sampler_relabel, pyg_random_walk_relabel
from GraphCLIP.generate_feature import encode_text_graph
from GraphCLIP.llm_prompts import eval_template
from exp.exp_downstream_tag import ExpDownstreamBatch
from root import ROOT_DIR


class ExpDownstreamBatchGraphCLIP(ExpDownstreamBatch):
    def __init__(self, args, pretrain_dict, name, dataset):
        super(ExpDownstreamBatchGraphCLIP, self).__init__(args, pretrain_dict, name, dataset)
        if args.compress_function == "none" :
            self.data = self._pure_data(self._build_encoded_data())

        self.class_texts = self._get_class_texts()

        # load subgraphs
        dir_path = f"{ROOT_DIR}/datasets/{self.args.target_data}/preprocess/graphclip/"
        os.makedirs(dir_path, exist_ok=True)
        sub_list_path = dir_path + f"sub_list({args.compress_function}).pt"
        
        self._print_main("Loading pre-generated subgraphs...")
        if os.path.exists(sub_list_path):
            self.sub_list = torch.load(sub_list_path)
        else:
            self.sub_list = self._build_subgraph_data(sub_list_path)

        self._print_main(f"Loaded pre-generated subgraphs from {sub_list_path}. And added PE feature.")

    def _get_class_texts(self):
        if self.task_name == 'edge':
            class_texts = [eval_template[self.args.target_data].format(c=c) for c in self.data.relation_texts]
        else:
            class_texts = [eval_template[self.args.target_data].format(c=c) for c in self.data.label_names]
            class_texts = [ti+desc for ti, desc in zip(class_texts, self.data.label_descs)]
        class_texts = [text.replace("_", " ") for text in class_texts]
        return class_texts
    

    def _get_pretrain_model(self, pretrain_model, strict=True):
        self.pretrain_setting = self._simple_pretrain_setting()
        path = (
                self.args.checkpoints + "/" + self.pretrain_setting + f"_{self.args.compress_function}"+ "/" + "checkpoint.pth"
        )

        self._load_checkpoint(path, pretrain_model, strict=strict)
        return pretrain_model
    
    def _build_encoded_data(self):
        data = encode_text_graph(
            data=self.data,
            name=self.args.target_data,
            lm_type="tiny",
            chunk_size=1024,
            device=self.device,
        )
        self._print_main(f"Finished generating node features for {self.args.target_data} dataset.")
        return data
    
    def _build_subgraph_data(self, sub_list_path):
        sub_list = []
        transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
        if hasattr(self.data, "batch") and self.data.batch is not None:
            for i in self.data.batch.unique():
                node_mask = (self.data.batch == i)
                node_indices = torch.where(node_mask)[0]
                sub_x = self.data.x[node_mask]
                sub_edge_index, _ = subgraph(
                    node_indices,
                    self.data.edge_index,
                    relabel_nodes=True,
                    num_nodes=self.data.num_nodes
                )
                sub_data = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=-1)
                sub_data = transform(sub_data) # add PE
                sub_list.append(sub_data)
        else:
            is_undirected = False if self.args.task_name == "edge" else True
            if self.args.sampler == "rw":
                sub_list = pyg_random_walk_relabel(torch.arange(self.data.num_nodes), self.data, 
                                                   length=self.args.walk_steps, restart_prob=self.args.restart,
                                                   is_undirected=is_undirected,transform=transform)
                
                # _, all_edges = pyg_random_walk(torch.arange(self.data.num_nodes), self.data, length=self.args.walk_steps, restart_prob=self.args.restart)
                # for id, edges in enumerate(all_edges):
                #     if edges.shape[1] == 0:
                #         edges = torch.tensor([[id],[id]])
                #     # reindex
                #     node_idx = torch.unique(edges)
                #     node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
                #     sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
                #     target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
                #     edge_index = torch.IntTensor([sources_idx, target_idx]).long()
                #     sub_data = Data(x=self.data.x[node_idx], edge_index=edge_index, root_n_index=node_idx_map[id])
                #     sub_data = transform(sub_data) # add PE
                #     sub_list.append(sub_data)
            
            elif self.args.sampler == 'khop':
                sub_list = ego_graphs_sampler_relabel(torch.arange(self.data.num_nodes), self.data, hop=self.args.k, transform=transform)
            else:
                raise ValueError(f"Unknown sampler type {self.args.sampler}")

        torch.save(sub_list, sub_list_path)
        return sub_list
    
    def _get_loader_subgraph(self, task):
        # batch: keep the original batch index
        task_ids = torch.tensor(task["idx"], dtype=torch.int64)
        task_labels = torch.tensor(task["labels"], dtype=torch.int64)

        persistent_workers=False
        task_dataset = TensorDataset(task_ids, task_labels)
        if self.args.task_name in ["node", "graph"]:
            loader = DataLoader(task_dataset,
                                batch_size=self.args.batch_size, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers=self.args.num_workers,
                                persistent_workers=persistent_workers)
            for ids, labels in loader:
                batch = Batch.from_data_list([self.sub_list[i.item()] for i in ids])
                yield batch, None , labels

        elif self.args.task_name == "edge":
            loader = DataLoader(task_dataset,
                                batch_size=self.args.batch_size, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers=self.args.num_workers,
                                persistent_workers=persistent_workers)
            for ids, labels in loader:
                batch_edge = self.data.edge_index[:, ids]
                unique_node_ids, edge_ids = torch.unique(
                    batch_edge,
                    sorted=False,
                    return_inverse=True
                )
                batch = Batch.from_data_list([self.sub_list[i.item()] for i in unique_node_ids])
                yield batch, edge_ids, labels

if __name__ == '__main__':
    from GraphCLIP.utils.args import Arguments
    from data_provider import *
    args = Arguments().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_exps = {
        "exp1": pretrain,
        "exp2": pretrain,
        "exp3": pretrain_exp3,
        "exp4": pretrain_exp4,
    }
    pretrain_dict = pretrain_exps[args.model_id]

    exps = {
        f"exp{i}": {
            "node": globals()[f"NC_exp{i}"],
            "edge": globals()[f"EC_exp{i}"],
            "graph": globals()[f"GC_exp{i}"],
        }
        for i in range(0, 5)
    }
    for name, dataset in exps[args.exp_id][args.task_name].items():
        exp = ExpDownstreamBatchGraphCLIP(args, pretrain_dict, name, dataset)