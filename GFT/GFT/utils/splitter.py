import numpy as np
import torch
from utils.others import idx2mask, mask2idx, sample_proto_instances_for_graph

# Get shared labels between train, valid, and test
def get_shared_labels(train_labels, valid_labels, test_labels, n_shot, n_query):
    base_shared_labels = np.intersect1d(
        np.unique(train_labels), np.unique(valid_labels)
    )
    base_shared_labels = np.intersect1d(base_shared_labels, np.unique(test_labels))

    shared_labels = []
    for label in base_shared_labels:
        if (
                np.sum(train_labels == label) >= n_shot
                and np.sum(valid_labels == label) >= n_query
                and np.sum(test_labels == label) >= n_query
        ):
            shared_labels.append(label)
    return shared_labels


# Randomly select n_way shared labels
def get_random_shared_labels(
        train_labels, valid_labels, test_labels, n_way, n_shot, n_query
):
    shared_labels = get_shared_labels(
        train_labels, valid_labels, test_labels, n_shot, n_query
    )
    n_way = min(n_way, len(shared_labels))
    ways = np.random.choice(shared_labels, n_way, replace=False)
    return ways


# For few-shot setting, select n_train samples for each class. The validation and test sets different.
# For zero-shot setting, does not select n_train samples for each class. The validation and test sets are the same.
# For in-context setting, does not select n_train samples for each class. The validation and test sets are the same.
def get_split(split, labels, params):
    labels = labels.cpu().numpy()
    setting = params["setting"]
    n_task, n_shot, n_way, n_query, n_train = (
        params["n_task"],
        params["n_shot"],
        params["n_way"],
        params["n_query"],
        params["n_train"],
    )
    num_samples = len(labels)
    idx_is_mask = split["train"].dtype == torch.bool

    train_idx, valid_idx, test_idx = (
        mask2idx(split["train"]) if idx_is_mask else split["train"],
        mask2idx(split["valid"]) if idx_is_mask else split["valid"],
        mask2idx(split["test"]) if idx_is_mask else split["test"],
    )
    train_labels, valid_labels, test_labels = (
        labels[train_idx],
        labels[valid_idx],
        labels[test_idx],
    )

    # Select finetune samples for "FEW-SHOT"
    final_train_idx = []
    if setting in ["few_shot"]:
        ways = get_shared_labels(
            train_labels, valid_labels, test_labels, n_train, n_query
        )
        way_indices = [np.where(labels == way)[0] for way in ways]
        for w_idx in way_indices:
            w_idx_train = np.intersect1d(w_idx, train_idx)
            final_train_idx.extend(
                np.random.choice(w_idx_train, n_train, replace=False)
            )
    elif setting in ["zero_shot", "in_context"]:
        print("Does not need to select samples for fine-tuning.")
    final_train_idx = idx2mask(final_train_idx, num_samples)

    # Select test set for "FEW-SHOT", "ZERO-SHOT", and "IN-CONTEXT"
    test_support_idx, test_query_idx = [], []
    for task in range(n_task):
        ways = get_random_shared_labels(
            train_labels, valid_labels, test_labels, n_way, n_shot, n_query
        )
        way_indices = [np.where(labels == way)[0] for way in ways]

        tmp_support_idx, tmp_query_idx = [], []
        for w_idx in way_indices:
            w_idx_support = np.intersect1d(w_idx, train_idx)
            w_idx_support = np.random.choice(w_idx_support, n_shot, replace=False)
            tmp_support_idx.extend(w_idx_support)

            w_idx_query = np.intersect1d(w_idx, test_idx)
            w_idx_query = np.random.choice(w_idx_query, n_query, replace=False)
            tmp_query_idx.extend(w_idx_query)
        test_support_idx.append(idx2mask(tmp_support_idx, num_samples))
        test_query_idx.append(idx2mask(tmp_query_idx, num_samples))

    if setting in ["few_shot"]:
        # The validation set is separate for few-shot learning
        valid_support_idx, valid_query_idx = [], []
        for task in range(n_task):
            ways = get_random_shared_labels(
                train_labels, valid_labels, test_labels, n_way, n_shot, n_query
            )
            way_indices = [np.where(labels == way)[0] for way in ways]

            tmp_support_idx, tmp_query_idx = [], []
            for w_idx in way_indices:
                w_idx_support = np.intersect1d(w_idx, train_idx)
                w_idx_support = np.random.choice(w_idx_support, n_shot, replace=False)
                tmp_support_idx.extend(w_idx_support)

                w_idx_query = np.intersect1d(w_idx, test_idx)
                w_idx_query = np.random.choice(w_idx_query, n_query, replace=False)
                tmp_query_idx.extend(w_idx_query)
            valid_support_idx.append(idx2mask(tmp_support_idx, num_samples))
            valid_query_idx.append(idx2mask(tmp_query_idx, num_samples))
    elif setting in ["zero_shot", "in_context"]:
        # The validation set is the same as the test set
        valid_support_idx = test_support_idx
        valid_query_idx = test_query_idx

    return {
        "train": final_train_idx,
        "valid": {"support": valid_support_idx, "query": valid_query_idx},
        "test": {"support": test_support_idx, "query": test_query_idx},
    }


def get_split_graph(split, labels, params):
    labels = labels.cpu().numpy()
    setting = params["setting"]
    n_task, n_shot, n_way, n_query, n_train = (
        params["n_task"],
        params["n_shot"],
        params["n_way"],
        params["n_query"],
        params["n_train"],
    )
    num_samples = len(labels)

    train_idx, valid_idx, test_idx = (
        split["train"],
        split["valid"],
        split["test"]
    )
    train_labels, valid_labels, test_labels = (
        labels[train_idx],
        labels[valid_idx],
        labels[test_idx],
    )

    # Select finetune samples for "FEW-SHOT"
    select_few_shot_samples = sample_proto_instances_for_graph
    if setting in ["few_shot"]:
        final_train_idx, final_train_labels = select_few_shot_samples(
            labels, train_idx, n_train
        )
    elif setting in ["zero_shot", "in_context"]:
        print("Does not need to select samples for fine-tuning.")

    # Select test set for "FEW-SHOT", "ZERO-SHOT", and "IN-CONTEXT"
    test_support_idx, test_query_idx = [], []
    test_support_label, test_query_label = [], []
    for task in range(n_task):
        tmp_support_idx, tmp_support_label = select_few_shot_samples(
            labels, train_idx, n_shot
        )
        tmp_query_idx, tmp_query_label = select_few_shot_samples(
            labels, test_idx, n_query
        )
        test_support_idx.append(tmp_support_idx)
        test_query_idx.append(tmp_query_idx)
        test_support_label.append(tmp_support_label)
        test_query_label.append(tmp_query_label)

    if setting in ["few_shot"]:
        # The validation set is separate for few-shot learning
        valid_support_idx, valid_query_idx = [], []
        valid_support_label, valid_query_label = [], []
        for task in range(n_task):
            tmp_support_idx, tmp_support_label = select_few_shot_samples(
                labels, train_idx, n_shot
            )
            tmp_query_idx, tmp_query_label = select_few_shot_samples(
                labels, test_idx, n_query
            )
            valid_support_idx.append(tmp_support_idx)
            valid_query_idx.append(tmp_query_idx)
            valid_support_label.append(tmp_support_label)
            valid_query_label.append(tmp_query_label)
    elif setting in ["zero_shot", "in_context"]:
        # The validation set is the same as the test set
        valid_support_idx, valid_query_idx = test_support_idx, test_query_idx
        valid_support_label, valid_query_label = test_support_label, test_query_label

    return {
        "train": [item for sublist in final_train_idx.values() for item in sublist],
        # Here we only update the train_idx
        "valid": {
            "support": {'idx': valid_support_idx, 'label': valid_support_label},
            "query": {'idx': valid_query_idx, 'label': valid_query_label}
        },
        "test": {
            "support": {'idx': test_support_idx, 'label': test_support_label},
            "query": {'idx': test_query_idx, 'label': test_query_label}
        },
    }
