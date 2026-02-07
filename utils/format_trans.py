import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.data import TemporalData, HeteroData, Data, InMemoryDataset, Batch
from collections import defaultdict


def temporal_to_hetero(temporaldata: TemporalData) -> HeteroData:
    hetero = HeteroData()

    src, dst, t, y = temporaldata.src.clone(), temporaldata.dst.clone(), temporaldata.t.clone(), temporaldata.y.clone()

    src_unique, src_inv = torch.unique(src, sorted=True, return_inverse=True)
    dst_unique, dst_inv = torch.unique(dst, sorted=True, return_inverse=True)

    num_src_nodes = src_unique.size(0)
    num_dst_nodes = dst_unique.size(0)

    edge_index = torch.stack([src_inv, dst_inv], dim=0)  # shape [2, num_edges]
    src_name = temporaldata.src_texts[0] if hasattr(temporaldata, 'src_texts') else 'src'
    dst_name = temporaldata.dst_texts[0] if hasattr(temporaldata, 'dst_texts') else 'dst'
    relation_name = temporaldata.relation_texts[0] if hasattr(temporaldata, 'relation_texts') else 'to'
    hetero[src_name, relation_name, dst_name].edge_index = edge_index
    hetero[dst_name, relation_name, src_name].edge_index = edge_index.flip(0)

    if hasattr(temporaldata, 't') and temporaldata.t is not None:
        hetero[src_name, relation_name, dst_name].edge_time = t
        hetero[dst_name, relation_name, src_name].edge_time = t

    if hasattr(temporaldata, 'msg') and temporaldata.msg is not None:
        hetero[src_name, relation_name, dst_name].edge_attr = temporaldata.msg
        hetero[dst_name, relation_name, src_name].edge_attr = temporaldata.msg

    latest_t = torch.full((num_src_nodes,), -1, dtype=torch.long)
    latest_y = torch.full((num_dst_nodes,), -1, dtype=y.dtype)

    for i in range(src_inv.size(0)):
        node = src_inv[i].item()
        if t[i] > latest_t[node]:
            latest_t[node] = t[i]
            latest_y[node] = y[i]

    hetero[src_name].x = torch.eye(num_src_nodes) if num_src_nodes <= 1000 else torch.eye(
        num_src_nodes).to_sparse()
    hetero[dst_name].x = torch.eye(num_dst_nodes) if num_dst_nodes <= 1000 else torch.eye(
        num_dst_nodes).to_sparse()

    hetero[src_name].y = latest_y

    if hasattr(temporaldata, 'n_id'):
        hetero[src_name].n_id = src_unique
        hetero[dst_name].n_id = dst_unique

    return hetero


def temporal_to_data(temporaldata: TemporalData, need_y=False) -> Data:
    data = Data()
    src, dst, t, y = temporaldata.src.clone(), temporaldata.dst.clone(), temporaldata.t.clone(), temporaldata.y.clone()

    src_unique = torch.unique(src, sorted=True)
    dst_unique = torch.unique(dst, sorted=True)
    all_nodes = torch.cat([src_unique, dst_unique])
    all_nodes_unique = torch.unique(all_nodes, sorted=True)
    num_nodes = all_nodes_unique.size(0)
    num_edges = src.size(0)
    msg = temporaldata.msg if hasattr(temporaldata, 'msg') else None
    # Node features
    data.x = torch.eye(num_nodes) if num_nodes <= 1000 else torch.eye(
        num_nodes).to_sparse()
    # Edge info
    edge_index = torch.stack([src, dst], dim=0)
    edge_index_rev = edge_index.flip(0)  # reverse direction
    data.edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    # Edge features
    if msg is not None:
        data.edge_attr = torch.cat([msg, msg], dim=0)
    src_name = temporaldata.src_texts[0] if hasattr(temporaldata, 'src_texts') else 'src'
    dst_name = temporaldata.dst_texts[0] if hasattr(temporaldata, 'dst_texts') else 'dst'
    relation_name = temporaldata.relation_texts[0] if hasattr(temporaldata, 'relation_texts') else 'to'

    data.node_type = torch.tensor([0 if i in src_unique.tolist() else 1 for i in all_nodes_unique.tolist()], dtype=torch.long)
    data.edge_type = torch.tensor([0] * num_edges + [1] * num_edges, dtype=torch.long)
    # Textual info
    src_set = set(src_unique.tolist())
    dst_set = set(dst_unique.tolist())
    data.raw_texts = [
        f"{src_name}/{dst_name}" if i in src_set and i in dst_set else
        src_name if i in src_set else
        dst_name
        for i in all_nodes_unique.tolist()
    ]
    data.relation_texts = [relation_name] + [f"rev_{relation_name}"]

    if need_y is True:
        # Node labels: latest timestamp
        latest_t = {}
        latest_y = {}
        for i in range(src.size(0)):
            node = src[i].item()
            if node not in latest_t or t[i] > latest_t[node]:
                latest_t[node] = t[i]
                latest_y[node] = y[i]

        node_labels = []
        for idx in all_nodes_unique.tolist():
            if idx in latest_y:
                node_labels.append(latest_y[idx].item())
            else:
                node_labels.append(-1)  # default -1 for nodes without label
        data.y = torch.tensor(node_labels, dtype=torch.long)
    
    if hasattr(temporaldata, 'label_names') and temporaldata.label_names is not None:
        data.label_names = temporaldata.label_names + ['N/A']
    if hasattr(temporaldata, 'label_descs') and temporaldata.label_descs is not None:
        data.label_descs = temporaldata.label_descs + [', which refers to nodes we do not use.']

    return data


def hetero_to_data(heterodata: HeteroData, need_y=False, use_embedding=False) -> Data:
    data = Data()
    node_offset = {}
    x_list = []
    raw_texts = []
    node_type_list = []
    current_node_id = 0
    main_node_type = None

    # one-hot/embedding for no feature nodes
    max_feat_dim = max(feat.size(1) for feat in heterodata.x_dict.values())
    embedding_layers = {}
    for node_type in heterodata.node_types:
        if hasattr(heterodata[node_type], 'y') and heterodata[node_type].y is not None:
            main_node_type = node_type  # in our datasets, only one node type has labels
        if hasattr(heterodata[node_type], 'x') and heterodata[node_type].x is not None:
            pass
        elif (not hasattr(heterodata[node_type], 'x') or heterodata[node_type].x is None) and hasattr(
                heterodata[node_type], 'num_nodes'):
            num_nodes = heterodata[node_type].num_nodes
            if use_embedding is True:
                embedding_layers[node_type] = nn.Embedding(num_nodes, max_feat_dim)
                heterodata[node_type].x = embedding_layers[node_type](
                    torch.arange(num_nodes, dtype=torch.long))
            else:
                heterodata[node_type].x = torch.eye(num_nodes)

        else:
            raise AttributeError('No x or num_nodes')

    if use_embedding is True:
        data.embedding_layers = embedding_layers

    # padding nodes
    max_feat_dim = max(feat.size(1) for feat in heterodata.x_dict.values())
    for idx, node_type in enumerate(heterodata.node_types):
        node_feat = heterodata[node_type].x
        feat_dim = node_feat.size(1)
        num_nodes = node_feat.size(0)

        if feat_dim < max_feat_dim:
            pad = max_feat_dim - feat_dim
            padded_feat = F.pad(node_feat, (0, pad), mode='constant', value=0)
        else:
            padded_feat = node_feat

        # record the starting index of each node type, which is used for edge_index adjustment
        node_offset[node_type] = current_node_id
        x_list.append(padded_feat)
        raw_texts.extend([node_type] * num_nodes)
        node_type_list.extend([idx] * num_nodes)
        current_node_id += num_nodes

    data.x = torch.cat(x_list, dim=0)
    if data.num_nodes > 10000 and data.x.size(1) > 1000:
        data.x = data.x.to_sparse()

    data.raw_texts = raw_texts  # List[str]
    data.node_type = torch.tensor(node_type_list)
    
    # edges
    final_edge_index = []
    final_edge_type = []
    final_relation_texts = ['' for _ in range(100)]  # assume less than 100 edge types

    edge_group = defaultdict(lambda: {"rels": [], "types": []})#defaultdict(lambda: {"rels": [], "triples": [], "types": []})
    current_type_id = 0

    for (src_type, rel_type, dst_type), edge_data in heterodata.edge_index_dict.items():
        src_offset = node_offset[src_type]
        dst_offset = node_offset[dst_type]

        edge_index = edge_data.clone()
        edge_index[0] += src_offset
        edge_index[1] += dst_offset

        if src_type == dst_type:
            # only group the edges with same src and dst type, as they are likely to have duplicated edges 
            # A1 -> A2 and A2 -> A1 can exist in one edge type.
            for i in range(edge_index.size(1)):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                edge_group[(u, v)]["rels"].append(rel_type)
                edge_group[(u, v)]["types"].append(current_type_id)
            current_type_id += 1
        else:
            # for edges with different src and dst type, we do not group them, as A -> B only have one  B -> A edge
            final_edge_index.extend(edge_index.t().tolist())
            final_edge_type.extend([current_type_id] * edge_index.size(1))
            final_relation_texts[current_type_id] = rel_type
            current_type_id += 1

    # process the grouped edges
    combo2id = {}
    new_type_id = current_type_id

    for (u, v), info in edge_group.items():
        final_edge_index.append([u, v])
        if len(info["types"]) == 1:
            final_edge_type.append(info["types"][0])
            final_relation_texts[info["types"][0]] = info["rels"][0]
        else:
            rel_combo = tuple(sorted(set(info["rels"])))
            if rel_combo not in combo2id:
                combo2id[rel_combo] = new_type_id
                new_type_id += 1
            merged_rel = ", ".join(rel_combo)
            final_edge_type.append(combo2id[rel_combo])
            final_relation_texts[new_type_id] = merged_rel

    data.edge_index = torch.tensor(final_edge_index, dtype=torch.long).t().contiguous()
    data.edge_type = torch.tensor(final_edge_type, dtype=torch.long)
    data.relation_texts = [text for text in final_relation_texts if text]
    data.edge_attr = torch.zeros((data.edge_index.size(1), 1), dtype=torch.float)

    if main_node_type is not None:
        if need_y is True:
            num_nodes_total = data.num_nodes
            main_y = heterodata[main_node_type].y
            y_full_shape = (num_nodes_total,)
            if len(main_y.shape) > 1:
                y_full_shape += main_y.shape[1:]

            y_full = torch.full(y_full_shape, -1, dtype=torch.long)  # default -1
            offset = node_offset[main_node_type]
            y_full[offset: offset + main_y.size(0)] = main_y
            data.y = y_full

        if hasattr(heterodata[main_node_type], 'label_names') and heterodata[main_node_type].label_names is not None:
            data.label_names = heterodata[main_node_type].label_names + ['N/A']
        if hasattr(heterodata[main_node_type], 'label_descs') and heterodata[main_node_type].label_descs is not None:
            data.label_descs = heterodata[main_node_type].label_descs + [', which refers to nodes we do not use.']

    return data


def multi_to_one(dataset: InMemoryDataset, need_y=False):
    data_list = []
    for data in dataset:
        data_list.append(data)
    new_data = Batch.from_data_list(data_list)

    if hasattr(dataset.data, 'label_names') and dataset.data.label_names is not None:
        new_data.label_names = dataset.data.label_names
    if hasattr(dataset.data, 'label_descs') and dataset.data.label_descs is not None:
        new_data.label_descs = dataset.data.label_descs
    if need_y is False:
        if hasattr(new_data, 'y'):
            del new_data.y
    return new_data


def one_to_multi(data):
    if isinstance(data, Batch):
        return data.to_data_list()
    
    data_list = []
    for i in data.batch.unique():
        node_mask = (data.batch == i)
        node_indices = torch.where(node_mask)[0]
        sub_x = data.x[node_mask]
        sub_edge_index, _ = subgraph(
            node_indices,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        sub_data = Data(x=sub_x, edge_index=sub_edge_index)
        data_list.append(sub_data)
    return data_list


def simple_process_hetero(heterodata: HeteroData, need_y=False, fill_onehot=True):
    for node_type in heterodata.node_types:
        store = heterodata[node_type]

        if not need_y:
            for key in ['y', 'train_mask', 'val_mask', 'test_mask']:
                if key in store:
                    del store[key]

        if fill_onehot and 'x' not in store:
            if hasattr(store, 'num_nodes') and store.num_nodes > 0:
                store.x = torch.eye(store.num_nodes)
                
    return heterodata


def simple_process_temporal(data: TemporalData, need_y=False):
    for key in ['label_names', 'label_descs', 'src_texts', 'dst_texts', 'relation_texts']:
        if key in data:
            del data[key]
    if not need_y:
        for key in ['y', 'train_mask', 'val_mask', 'test_mask']:
            if key in data:
                del data[key]
    if not 'e_id' in data:
        data.e_id = torch.arange(data.num_edges)
                
    return data


if __name__ == '__main__':
    from data_provider.data_generator import hiv
    print(multi_to_one(hiv()))
    print(torch.unique((hiv()).y))
