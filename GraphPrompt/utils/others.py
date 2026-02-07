from torch_geometric.data import Data
import torch

def create_x(data: Data) -> Data:
    '''create node features by one hot if not exist'''
    if hasattr(data, "x") and data.x is not None:
        return data
    elif hasattr(data, "num_nodes") and data.num_nodes is not None:
        data.x = torch.eye(data.num_nodes) if data.num_nodes <= 1000 else torch.eye(
            data.num_nodes).to_sparse()  # torch.randn(data.num_nodes, f_size)
        return data
    else:
        raise AttributeError("No x or num_nodes")


def complete_data(data: Data, name, need_y=False) -> Data:
    '''complete data attributes'''
    new_data = Data()
    new_data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    new_data.edge_index = data.edge_index
    new_data.name = name
    if not hasattr(data, 'edge_type') or data.edge_type is None:
        new_data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
    else:
        new_data.edge_type = data.edge_type

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        new_data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float)
    else:
        new_data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

    if not hasattr(data, 'raw_texts') or data.raw_texts is None:
        new_data.raw_texts = ['N/A'] * data.num_nodes
    else:
        new_data.raw_texts = data.raw_texts

    if not hasattr(data, 'relation_texts') or data.relation_texts is None:
        new_data.relation_texts = ['to']
    else:
        new_data.relation_texts = data.relation_texts

    if hasattr(data, 'batch') and data.batch is not None:
        new_data.batch = data.batch

    if need_y and hasattr(data, 'y') and data.y is not None:
        if not hasattr(data, 'y') or data.y is None:
            raise ValueError(f"Data {name} has no labels 'y' but need_y is True.")
        else:
            new_data.y = data.y
            
            if hasattr(data, 'label_names') and data.label_names is not None:
                new_data.label_names = data.label_names

            if hasattr(data, 'label_descs') and data.label_descs is not None:
                new_data.label_descs = data.label_descs

    return new_data