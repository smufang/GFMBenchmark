from dataset.process_datasets import span_node_and_edge_idx, filter_unnecessary_attrs


def pre_node(dataset):
    dataset = span_node_and_edge_idx(dataset)
    dataset = filter_unnecessary_attrs(dataset)
    return dataset


def pre_link(dataset):
    dataset = span_node_and_edge_idx(dataset)
    dataset = filter_unnecessary_attrs(dataset, mode="finetune")
    return dataset


def pre_graph(dataset):
    return dataset
