from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch
import torch.nn.functional as F

# from data_provider.multigraph_loader import MultiGraphLoader
from data_provider.maxnode_loader import MultiGraphLoader
from data_provider.subbatch_loader import BatchGraphLoader
from data_provider import *
from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
from itertools import chain
from typing import List, Dict, Union
import warnings
import ftfy
import os
from root import ROOT_DIR

warnings.filterwarnings("ignore")


def create_x(data: Data) -> Data:
    """create node features by one hot if not exist"""
    def sparse_one_hot(num_nodes):
        idx = torch.arange(num_nodes, dtype=torch.long)
        indices = torch.stack([idx, idx])  # 2 x N
        values = torch.ones(num_nodes)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    if getattr(data, "x", None) is not None:
        return data
    elif getattr(data, "num_nodes", None) is not None:
        data.x = (
            torch.eye(data.num_nodes)
            if data.num_nodes <= 1000
            else sparse_one_hot(data.num_nodes)
        )  # torch.randn(data.num_nodes, f_size)
        return data
    else:
        raise AttributeError("No x or num_nodes")


def complete_data(data: Data, name, need_y=False, need_token_cache=False) -> Data:
    """complete data attributes, including:
    - x              Tensor([num_nodes, num_features])
    - edge_index     Tensor([2, num_edges])
    - name           Str
    - batch          Tensor([num_nodes])                       (only for graph-level)
    - edge_type      Tensor([num_edges])
    - node_type      Tensor([num_nodes])
    - edge_attr      Tensor([num_edges, num_edge_features])
    - raw_texts      List[num_nodes]
    - relation_texts List[num_edge_types]
    - y              Different based on Tasks and Data format  (only for need_y=True)                 
    - label_names    List[num_classes]                         (if has)
    - label_descs    List[num_classes]                         (if has)
    - token_cache    Tensor([num_nodes, num_features])         (only for need_token_cache=True)
    """
    new_data = Data()
    new_data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    new_data.edge_index = data.edge_index
    new_data.name = name
    if getattr(data, "edge_type", None) is None:
        new_data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
    else:
        new_data.edge_type = data.edge_type

    if getattr(data, "node_type", None) is None:
        new_data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    else:
        new_data.node_type = data.node_type

    if getattr(data, "edge_attr", None) is None:
        new_data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float)
    else:
        new_data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

    if getattr(data, "raw_texts", None) is None:
        new_data.raw_texts = ["N/A"] * data.num_nodes
    else:
        new_data.raw_texts = [
            ftfy.fix_text(text) for text in data.raw_texts
        ]  # fix text encoding issues

    if getattr(data, "relation_texts", None) is None:
        new_data.relation_texts = ["to"] * data.num_edge_types # Align with data.num_edge_types to prevent edge_type index overflow.
    else:
        new_data.relation_texts = [
            ftfy.fix_text(text) for text in data.relation_texts
        ]  # fix text encoding issues

    if getattr(data, "batch", None) is not None:
        new_data.batch = data.batch

    if need_y:
        if getattr(data, "y", None) is None:
            raise ValueError(f"Data {name} has no labels 'y' but need_y is True.")
        else:
            new_data.y = data.y

    if getattr(data, "label_names", None) is not None:
        new_data.label_names = data.label_names

    if getattr(data, "label_descs", None) is not None:
        new_data.label_descs = data.label_descs

    if need_token_cache:
        # G2P2 tokenizer cache
        dir_path = f"{ROOT_DIR}/datasets/{name}/preprocess"
        os.makedirs(dir_path, exist_ok=True)
        path = dir_path + "/token_cache_g2p2.pt"
        if not os.path.exists(path):
            from G2P2.model import tokenize

            token_cache = tokenize(new_data.raw_texts, context_length=128)
            torch.save(token_cache, path)
            print(f"Saved token cache for dataset {name} to {path}.")
        new_data.token_cache = torch.load(path)

    return new_data


def get_original_data(
    generators, is_text=False, need_y=False, need_token_cache=False
) -> Dict[str, Data]:
    """create original data cache"""

    def _pure_data(_data, need_y=False):
        new_data = Data()
        new_data.x = _data.x
        new_data.edge_index = _data.edge_index
        if need_y:
            new_data.y = _data.y
        return new_data

    original_dict = {}
    for name, generator in generators.items():
        dataset = generator()
        if len(dataset) == 1:
            data = dataset[0]
            if isinstance(data, Data):
                data = create_x(data)
            elif isinstance(data, HeteroData):
                data = hetero_to_data(data, need_y=need_y)
            elif isinstance(data, TemporalData):
                data = temporal_to_data(data, need_y=need_y)
            else:
                raise TypeError(f"Unsupported data format ({name}): {type(data)}")
        elif len(dataset) > 1:
            data = multi_to_one(dataset, need_y=need_y)
            data = create_x(data) # for ogbg-ppa
        else:
            raise ValueError(f"Dataset {name} length < 1: {len(dataset)}")
        data = complete_data(
            data, name, need_y=need_y, need_token_cache=need_token_cache
        )
        if not is_text:
            original_dict[name] = _pure_data(data, need_y=need_y)
        else:
            original_dict[name] = data
    return original_dict


def get_compressed_data(
    generators, compress_fc, k=50, need_y=False, need_token_cache=False
) -> Dict[str, Data]:
    """create compressed data cache"""
    compressed_dict = {}
    for name, generator in generators.items():
        dataset = generator()
        if len(dataset) == 1:
            data = dataset[0]
            if isinstance(data, Data):
                data = create_x(data)
            elif isinstance(data, HeteroData):
                data = hetero_to_data(data, need_y=need_y)
            elif isinstance(data, TemporalData):
                data = temporal_to_data(data, need_y=need_y)
            else:
                raise TypeError(f"Unsupported data format ({name}): {type(data)}")
            try:
                data.x = compress_fc(data.x, k)
            except:
                raise ValueError(
                    f"Compression function failed for dataset {name} with k={k}. "
                    f"Please check the compress_fc function."
                )
        elif len(dataset) > 1:
            data = multi_to_one(dataset, need_y=need_y)
            data = create_x(data) # for ogbg-ppa
            data.x = compress_fc(data.x, k)
        else:
            raise ValueError(f"Dataset {name} length < 1: {len(dataset)}")
        data = complete_data(
            data, name, need_y=need_y, need_token_cache=need_token_cache
        )
        compressed_dict[name] = data
    return compressed_dict


def data_sampler(
    data: Data,
    name,
    num_neighbors=(10, 5),
    batch_size=16,
    num_workers=0,
    need_input_nodes=False,
):
    data = create_x(data)

    x = (
        data.x
        if not (data.x.layout == torch.sparse_csr or data.x.layout == torch.sparse_coo)
        else data.x.to_dense()
    )  # Ensure x is dense for NeighborLoader
    # basic
    raw_texts = getattr(data, "raw_texts", None)
    edge_type = getattr(data, "edge_type", None)
    edge_attr = getattr(data, "edge_attr", None)
    relation_texts = getattr(data, "relation_texts", None)
    # extra
    token_cache = getattr(data, "token_cache", None)
    # control the input nodes for big graphs
    #if not need_input_nodes:
    if getattr(data, "batch", None) is not None:
        input_nodes = []
        batch_ids = data.batch.unique()
        for b in batch_ids:
            node_idx = (data.batch == b).nonzero(as_tuple=True)[0]
            num_sample = len(node_idx) // 16 + 1
            sampled = node_idx[torch.randperm(len(node_idx))[:num_sample]]
            input_nodes.extend(sampled.tolist())
        input_nodes = torch.tensor(input_nodes)
    else:
        num_sample = min(data.num_nodes**2 // data.num_edges + 1, data.num_nodes)
        sampled = torch.randperm(data.num_nodes)[:num_sample]
        input_nodes = sampled
    # else:
    #     input_nodes = None

    loader = NeighborLoader(
        Data(
            x=x,
            edge_index=data.edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            token_cache=token_cache,
        ),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=True,
        replace=False,
        directed=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    for subgraph in loader:
        out_subgraph = Data(x=subgraph.x, edge_index=subgraph.edge_index)
        node_ids = subgraph.n_id.tolist()

        if edge_type is not None:
            out_subgraph.edge_type = subgraph.edge_type

        if edge_attr is not None:
            out_subgraph.edge_attr = subgraph.edge_attr

        if raw_texts is not None:
            out_subgraph.raw_texts = [raw_texts[idx] for idx in node_ids]

        if relation_texts is not None and subgraph.edge_type is not None:
            # Here the relation_texts is mapping to each edge from edge_type
            out_subgraph.relation_texts = relation_texts

        out_subgraph.name = name

        if need_input_nodes:
            out_subgraph.input_nodes = torch.arange(
                len(subgraph.input_id), dtype=torch.long
            )

        if token_cache is not None:
            out_subgraph.token_cache = subgraph.token_cache
        del subgraph
        yield out_subgraph


def multidata_sampler(
    data: Data, name, batch_size=16, num_workers=0, need_input_nodes=False
):
    # This method is for multiple small graphs been merged into a single large graph using Batch.from_data_list()
    data = create_x(data)
    x = (
        data.x
        if not (data.x.layout == torch.sparse_csr or data.x.layout == torch.sparse_coo)
        else data.x.to_dense()
    )  # Ensure x is dense for NeighborLoader

    loader = BatchGraphLoader(
        data,
        graph_label_index=None,
        graph_label=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for subgraph in loader:
        subgraph.name = name
        # wiew all the nodes as input nodes
        if need_input_nodes:
            subgraph.input_nodes = torch.arange(subgraph.num_nodes, dtype=torch.long)
        # this is needed when testing, so delete in pretraining
        del subgraph.batch
        del subgraph.graph_label_index
        yield subgraph


def pretrain_sampler(
    generators,
    compressed_data=None,
    batch_size=1024,
    num_neighbors=(10, 10, 10, 10),
    num_workers=0,
    need_input_nodes=False,
) -> List[Data]:
    """
    Sample and process graphs for pretraining.

    Args:
        generators: Dictionary of dataset generators
        compressed_data: Pre-compressed data cache (optional)
        batch_size: The number of samples per batch
        num_neighbors: Tuple specifying the number of neighbors to sample at each layer
        num_workers: The number of sub-processes for data loading
        need_input_nodes: Whether to include input node indices in the output data
    """
    proc_graphs = []
    if compressed_data is None:
        for name, generator in generators.items():
            dataset = generator()
            if len(dataset) == 1:
                data = dataset[0]
                if isinstance(data, Data):
                    data = create_x(data)
                elif isinstance(data, HeteroData):
                    data = hetero_to_data(data)
                elif isinstance(data, TemporalData):
                    data = temporal_to_data(data)
                else:
                    raise TypeError(f"Unsupported data format ({name}): {type(data)}")
                for sub_data in data_sampler(
                    data,
                    name,
                    num_neighbors=num_neighbors,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    need_input_nodes=need_input_nodes,
                ):
                    proc_graphs.append(sub_data)
            elif len(dataset) > 1:
                data = multi_to_one(dataset)
                # as g2p2 using compressed_cache, so here we didn't modify as follow
                for sub_data in multidata_sampler(
                    data,
                    name,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    need_input_nodes=need_input_nodes,
                ):
                    proc_graphs.append(sub_data)
            else:
                raise ValueError(f"Dataset {name} length < 1: {len(dataset)}")
        return proc_graphs
    else:
        for name in generators.keys():
            data = compressed_data[name]
            if getattr(data, "batch", None) is not None and need_input_nodes is False:
                # g2p2 don't use this branch
                for sub_data in multidata_sampler(
                    data,
                    name,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    need_input_nodes=need_input_nodes,
                ):
                    proc_graphs.append(sub_data)
            else:
                for sub_data in data_sampler(
                    data,
                    name,
                    num_neighbors=num_neighbors,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    need_input_nodes=need_input_nodes,
                ):
                    proc_graphs.append(sub_data)
        return proc_graphs


def general_loader(
    sampler: List[Union[Data]], max_nodes, num_workers=0, compress_fc=None, k=50
) -> MultiGraphLoader:
    def _pad_edge_attr(data_list: List[Union[Data]]) -> List[Union[Data]]:
        # edge attr alignment by padding
        if getattr(data_list[0], "edge_attr", None) is None:
            return data_list
        
        max_edge_dim = max(
            (
                data.edge_attr.size(1)
                for data in data_list
            ),
            default=1,
        )

        for data in data_list:
            # alignment edge feature
            pad_edge = max_edge_dim - data.edge_attr.size(1)
            if pad_edge > 0:
                data.edge_attr = F.pad(
                    data.edge_attr, (0, pad_edge), mode="constant", value=0
                )
        return data_list

    def _re_idx_edge_type(data_list: List[Union[Data]]) -> List[Union[Data]]:
        # Re-index each data.item’s edge_type by adding the current offset,
        # so that edge_type values across the list are globally unique and align with the concatenated relation_texts.
        if getattr(data_list[0], "edge_type", None) is not None:
            curr_type_offset = 0
            for data in data_list:
                data.edge_type = data.edge_type + curr_type_offset
                curr_type_offset += len(data.relation_texts)
        return data_list

    def _re_idx_node_type(data_list: List[Union[Data]]) -> List[Union[Data]]:
        # Re-index each data.item’s node_type by adding the current offset,
        # so that node_type values across the list are globally unique.
        if getattr(data_list[0], "node_type", None) is not None:
            curr_type_offset = 0
            for data in data_list:
                data.node_type = data.node_type + curr_type_offset
                curr_type_offset += data.node_type.max().item() + 1
        return data_list

    def _compress_features(data_list: List[Union[Data]]) -> List[Union[Data]]:
        if compress_fc is not None:
            for data in data_list:
                data.x = compress_fc(data.x, k=k)
        return data_list

    loader = MultiGraphLoader(sampler, max_nodes, shuffle=True, num_workers=num_workers)
    for data_list in loader:
        # Following three functions actually don't used in current experiments
        data_list = _pad_edge_attr(data_list)
        data_list = _re_idx_edge_type(data_list)
        data_list = _re_idx_node_type(data_list)

        data_list = _compress_features(data_list)
        data = Batch.from_data_list(data_list)
        if getattr(data, "raw_texts", None) is not None:
            # search raw_texts of node 'n[i]' by data.raw_texts[i]
            data.raw_texts = list(chain.from_iterable(data.raw_texts))
        if getattr(data, "relation_texts", None) is not None:
            # search relation_texts of edge 'e[i]' by data.relation_texts[data.edge_type[i]]
            data.relation_texts = list(chain.from_iterable(data.relation_texts))
        if getattr(data, "name", None) is not None:
            data.name = [data.name[i] for i in data.batch]
        if getattr(data, "input_nodes", None) is not None:
            # adjust input_nodes index after batching
            input_nodes = []
            cum_nodes = 0
            for i in range(len(data_list)):
                sub_data = data_list[i]
                input_nodes.extend((sub_data.input_nodes + cum_nodes).tolist())
                cum_nodes += sub_data.num_nodes
            data.input_nodes = torch.tensor(input_nodes, dtype=torch.long)
        yield data


def pretrain_loader(
    pretrain_dict,
    num_neighbors=(10, 10, 10, 10),
    max_nodes=80000,
    batch_size=1024,
    num_workers=0,
    compress_fc=None,
    k=50,
    compressed_data=None,
    need_input_nodes=False,
):
    samplers = pretrain_sampler(
        pretrain_dict,
        compressed_data=compressed_data,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        num_workers=num_workers,
        need_input_nodes=need_input_nodes,
    )
    if compressed_data is None:
        # compress after sampler
        yield from general_loader(
            samplers, max_nodes, num_workers=num_workers, compress_fc=compress_fc, k=k
        )
    else:
        # compress before sampler
        yield from general_loader(
            samplers, max_nodes, num_workers=num_workers, compress_fc=None
        )


# def pretrain_loader(pretrain_dict, max_nodes=50000, num_workers=0, compress_fc=None, k=50, compressed_data=None):
#     if compressed_data is not None:
#         for name, data in compressed_data.items():
#             new_data = Data()
#             new_data.x = data.x
#             new_data.edge_index = data.edge_index
#             new_data.name = [name] * data.num_nodes
#             yield new_data

if __name__ == "__main__":
    from data_provider.data_generator import *
    from data_provider import pretrain
    from utils.compress_func import compress_pca

    compressed_data_ = get_compressed_data(pretrain, compress_fc=compress_pca, k=50)
    pretrain_loader = pretrain_loader(
        pretrain,
        max_nodes=80000,
        compress_fc=compress_pca,
        k=50,
        compressed_data=compressed_data_,
    )
    for i in range(10):
        count = 0
        for batch in pretrain_loader:
            count += 1
            print(batch)
            print(set(batch.name))
            print(count)
