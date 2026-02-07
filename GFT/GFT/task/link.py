import os

os.sys.path.append(os.path.join(os.path.abspath(""), "../../..", "data", "OneForAll"))

import numpy as np
import torch
from torch_geometric.loader import LinkNeighborLoader

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model, sample_proto_instances, mask2idx


def ft_link(model, dataset, loader, optimizer, split, labels, params, scheduler=None, **kwargs):
    model.train()

    device = get_device_from_model(model)
    setting = params["setting"]
    num_classes = params["num_classes"]
    query_node_code_first = params["query_node_code_first"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    proto_loss = torch.tensor(0.0).to(device)
    act_loss = torch.tensor(0.0).to(device)

    mini_batch = loader is not None

    if not mini_batch:
        # Encode

        x = dataset.node_text_feat[dataset.x]
        edge_index = dataset.edge_index
        edge_attr = dataset.edge_text_feat[dataset.xe]
        y = labels

        z = model.encode(x, edge_index, edge_attr)

        # Compute edge embeddings

        train_mask = split["train"]
        edge_index_train, y_train = edge_index[:, train_mask], y[train_mask]
        edge_z_train = (z[edge_index_train[0]] + z[edge_index_train[1]]) / 2

        if use_proto_clf:
            # Compute Prototypes
            if query_node_code_first:
                # Case 1: Use node code to form edge code
                code, commit_loss = model.get_codes(z, use_orig_codes=True)
                edge_code_train = (code[edge_index_train[0]] + code[edge_index_train[1]]) / 2
            else:
                # Case 2: Query edge code using edge embeddings directly
                # This is the default case.
                edge_code_train, commit_loss = model.get_codes(edge_z_train, use_orig_codes=True)

            proto_emb = model.get_class_prototypes(edge_code_train, y_train, num_classes).detach()
            query_emb = edge_z_train if params['use_z_in_predict'] else edge_code_train  # Use train set

            # Compute Losses

            proto_loss = model.compute_proto_loss(query_emb, proto_emb, y_train) * params["lambda_proto"]

        if use_lin_clf:
            act_loss = model.compute_activation_loss(edge_z_train, y_train) * params["lambda_act"]

        loss = proto_loss + act_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        return {
            'proto_loss': proto_loss.item(),
            'act_loss': act_loss.item(),
            'loss': loss.item(),
        }
    else:
        if use_proto_clf:
            # Get prototype loader
            if setting == "standard":

                # For standard setting, we sample instances from
                # the training set to form prototypes.

                proto_idx = sample_proto_instances(labels, mask2idx(split["train"]),
                                                   num_instances_per_class=params['num_instances_per_class'])
                proto_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, proto_idx],
                    edge_label=labels[proto_idx],
                    batch_size=1024,
                    num_workers=8,
                )
            elif setting in ["few_shot"]:
                # In few-shot setting, we can directly use all instances
                # As the number of instances is small.
                proto_loader = loader

            # Encode Prototypes

            code_list, y_list = [], []
            for batch in proto_loader:
                batch = batch.to(device)

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]
                edge_label_index = batch.edge_label_index
                y = batch.edge_label

                z = model.encode(x, edge_index, edge_attr)
                edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                if query_node_code_first:
                    # Case 1: Use node code to form edge code
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                else:
                    # Case 2: Query edge code using edge embeddings directly
                    edge_code, _ = model.get_codes(edge_z, use_orig_codes=True)

                code_list.append(edge_code.detach())
                y_list.append(y)

            edge_code = torch.cat(code_list, dim=0)
            y = torch.cat(y_list, dim=0)
            proto_emb = model.get_class_prototypes(edge_code, y, num_classes)

        # Start Training

        total_proto_loss = 0
        total_act_loss = 0
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            # Encode
            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat[batch.xe]
            edge_label_index = batch.edge_label_index
            y = batch.edge_label

            z = model.encode(x, edge_index, edge_attr)
            edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

            if use_proto_clf:
                if query_node_code_first:
                    # Case 1: Use node code to form edge code
                    code, commit_loss = model.get_codes(z, use_orig_codes=True)
                    edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                else:
                    # Case 2: Query edge code using edge embeddings directly
                    edge_code, commit_loss = model.get_codes(edge_z, use_orig_codes=True)

                query_emb = edge_z if params['use_z_in_predict'] else edge_code  # Use train set
                proto_loss = model.compute_proto_loss(query_emb, proto_emb, y) * params["lambda_proto"]

            if use_lin_clf:
                act_loss = model.compute_activation_loss(edge_z, y) * params["lambda_act"]
            loss = proto_loss + act_loss

            total_proto_loss += proto_loss.item()
            total_act_loss += act_loss.item()
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        return {
            'proto_loss': total_proto_loss / len(loader),
            'act_loss': total_act_loss / len(loader),
            'loss': total_loss / len(loader),
        }


def eval_link(model, dataset, loader, split, labels, params, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]
    num_classes = params["num_classes"]
    query_node_code_first = params["query_node_code_first"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    pred_proto = 0
    pred_lin = 0

    mini_batch = loader is not None
    if not mini_batch:
        # Encode

        x = dataset.node_text_feat[dataset.x]
        edge_index = dataset.edge_index
        edge_attr = dataset.edge_text_feat[dataset.xe]
        y = labels

        z = model.encode(x, edge_index, edge_attr)

        if setting == "standard":

            train_mask = split["train"]
            edge_index_train, y_train = edge_index[:, train_mask], y[train_mask]
            edge_z = (z[edge_index[0]] + z[edge_index[1]]) / 2

            if use_proto_clf:
                # Compute Prototypes

                if query_node_code_first:
                    # Case 1: Query node code first and then form edge code
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    edge_code = (code[edge_index[0]] + code[edge_index[1]]) / 2
                    edge_code_train = edge_code[train_mask]
                else:
                    # Case 2: Query edge code directly
                    edge_code, _ = model.get_codes(edge_z, use_orig_codes=True)
                    edge_code_train = edge_code[train_mask]

                proto_emb = model.get_class_prototypes(edge_code_train, y_train, num_classes).detach()
                query_emb = edge_z if model.use_z_in_predict else edge_code  # Use all instances

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

            if use_lin_clf:
                pred_lin = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)

            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            # Evaluate

            train_mask, val_mask, test_mask = (split["train"], split["valid"], split["test"])
            train_value = evaluate(pred, y, train_mask, params)
            val_value = evaluate(pred, y, val_mask, params)
            test_value = evaluate(pred, y, test_mask, params)

            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[params['task']]
            }

        elif setting == "few_shot":
            n_task = len(split["valid"]["support"])
            train_values, val_values, test_values = [], [], []

            # Validation: few-shot, zero-shot, in-context
            # For zero-shot and in-context, the validation is the same as test.
            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                # Encode edge embedding

                edge_index_support, y_support = edge_index[:, s_mask], y[s_mask]
                edge_index_query, y_query = edge_index[:, q_mask], y[q_mask]
                edge_z_support = (z[edge_index_support[0]] + z[edge_index_support[1]]) / 2
                edge_z_query = (z[edge_index_query[0]] + z[edge_index_query[1]]) / 2

                if use_proto_clf:
                    # Compute edge prototypes

                    if query_node_code_first:
                        # Case 1: Query node code first and then form edge code
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        edge_code_support = (code[edge_index_support[0]] + code[edge_index_support[1]]) / 2
                        edge_code_query = (code[edge_index_query[0]] + code[edge_index_query[1]]) / 2
                    else:
                        # Case 2: Query edge code directly
                        edge_code_support, _ = model.get_codes(edge_z_support, use_orig_codes=True)
                        edge_code_query, _ = model.get_codes(edge_z_query, use_orig_codes=True)

                    proto_emb = model.get_class_prototypes(edge_code_support, y_support, num_classes).detach()
                    query_emb = edge_z_query if model.use_z_in_predict else edge_code_query

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(edge_z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, params=params)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                # Encode edge embedding

                edge_index_support, y_support = edge_index[:, s_mask], y[s_mask]
                edge_index_query, y_query = edge_index[:, q_mask], y[q_mask]
                edge_z_support = (z[edge_index_support[0]] + z[edge_index_support[1]]) / 2
                edge_z_query = (z[edge_index_query[0]] + z[edge_index_query[1]]) / 2

                if use_proto_clf:
                    # Compute edge prototypes

                    if query_node_code_first:
                        # Case 1: Query node code first and then form edge code
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        edge_code_support = (code[edge_index_support[0]] + code[edge_index_support[1]]) / 2
                        edge_code_query = (code[edge_index_query[0]] + code[edge_index_query[1]]) / 2
                    else:
                        # Case 2: Query edge code directly
                        edge_code_support, _ = model.get_codes(edge_z_support, use_orig_codes=True)
                        edge_code_query, _ = model.get_codes(edge_z_query, use_orig_codes=True)

                    proto_emb = model.get_class_prototypes(edge_code_support, y_support, num_classes).detach()
                    query_emb = edge_z_query if model.use_z_in_predict else edge_code_query

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(edge_z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, params=params)
                test_values.append(value)

            return {
                'train': np.mean(train_values),
                'val': np.mean(val_values),
                'test': np.mean(test_values),
                'metric': task2metric[params['task']]
            }
    else:
        if setting == "standard":

            if use_proto_clf:
                # Get prototype loader
                # Only works in standard setting
                proto_idx = sample_proto_instances(labels, mask2idx(split["train"]),
                                                   num_instances_per_class=params['num_instances_per_class'])

                proto_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, proto_idx],
                    edge_label=labels[proto_idx],
                    batch_size=1024,
                    num_workers=8,
                )

                # Encode Prototypes

                code_list, y_list = [], []
                for batch in proto_loader:
                    batch = batch.to(device)

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]
                    y = batch.edge_label
                    edge_label_index = batch.edge_label_index

                    z = model.encode(x, edge_index, edge_attr)
                    edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                    if query_node_code_first:
                        # Case 1: Use node code to form edge code
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                    else:
                        # Case 2: Query edge code using edge embeddings directly
                        edge_code, _ = model.get_codes(edge_z, use_orig_codes=True)

                    code_list.append(edge_code.detach())
                    y_list.append(y)

                edge_code = torch.cat(code_list, dim=0)
                y = torch.cat(y_list, dim=0)
                proto_emb = model.get_class_prototypes(edge_code, y, num_classes)

            # Prediction

            pred_list, y_list = [], []
            for batch in loader:
                batch = batch.to(device)

                # Encode
                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]
                edge_label_index = batch.edge_label_index
                y = batch.edge_label

                z = model.encode(x, edge_index, edge_attr)
                edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                if use_proto_clf:
                    if query_node_code_first:
                        # Case 1: Use node code to form edge code
                        code, commit_loss = model.get_codes(z, use_orig_codes=True)
                        edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                    else:
                        # Case 2: Query edge code using edge embeddings directly
                        edge_code, commit_loss = model.get_codes(edge_z, use_orig_codes=True)

                    query_emb = edge_z if model.use_z_in_predict else edge_code  # Use train set

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                pred_list.append(pred.detach())
                y_list.append(y)

            # Evaluate

            pred = torch.cat(pred_list, dim=0)
            y = torch.cat(y_list, dim=0)

            train_mask, val_mask, test_mask = (
                split["train"],
                split["valid"],
                split["test"],
            )
            train_value = evaluate(pred, y, train_mask, params)
            val_value = evaluate(pred, y, val_mask, params)
            test_value = evaluate(pred, y, test_mask, params)

            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[params['task']]
            }

        elif setting == "few_shot":
            n_task = len(split["valid"]["support"])
            train_values, val_values, test_values = [], [], []

            # Validation: few-shot, zero-shot, in-context
            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                # Define loaders for support and query sets
                proto_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, s_mask],
                    edge_label=labels[s_mask],
                    batch_size=1024,
                    num_workers=8,
                )
                query_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, q_mask],
                    edge_label=labels[q_mask],
                    batch_size=1024,
                    num_workers=8,
                )

                if use_proto_clf:

                    # Construct prototypes based on support set

                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]
                        edge_label_index = batch.edge_label_index
                        y = batch.edge_label

                        z = model.encode(x, edge_index, edge_attr)
                        edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                        if query_node_code_first:
                            # Case 1: Use node code to form edge code
                            code, _ = model.get_codes(z, use_orig_codes=True)
                            edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                        else:
                            # Case 2: Query edge code using edge embeddings directly
                            edge_code, _ = model.get_codes(edge_z, use_orig_codes=True)

                        code_list.append(edge_code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes)

                # Compute logits

                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]
                    edge_label_index = batch.edge_label_index
                    y = batch.edge_label

                    z = model.encode(x, edge_index, edge_attr)
                    edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                    if use_proto_clf:
                        if query_node_code_first:
                            # Case 1: Use node code to form edge code
                            code, commit_loss = model.get_codes(z, use_orig_codes=True)
                            edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                        else:
                            # Case 2: Query edge code using edge embeddings directly
                            edge_code, commit_loss = model.get_codes(edge_z, use_orig_codes=True)

                        query_emb = edge_z if model.use_z_in_predict else edge_code

                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)

                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                    pred_list.append(pred.detach())
                    y_list.append(y)

                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

                value = evaluate(pred, y, params=params)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                # Define loaders for support and query sets
                proto_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, s_mask],
                    edge_label=labels[s_mask],
                    batch_size=1024,
                    num_workers=8,
                )
                query_loader = LinkNeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    edge_label_index=dataset.edge_index[:, q_mask],
                    edge_label=labels[q_mask],
                    batch_size=1024,
                    num_workers=8,
                )

                if use_proto_clf:
                    # Construct prototypes based on support set

                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]
                        edge_label_index = batch.edge_label_index
                        y = batch.edge_label

                        z = model.encode(x, edge_index, edge_attr)
                        edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                        if query_node_code_first:
                            # Case 1: Use node code to form edge code
                            code, _ = model.get_codes(z, use_orig_codes=True)
                            edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                        else:
                            # Case 2: Query edge code using edge embeddings directly
                            edge_code, _ = model.get_codes(edge_z, use_orig_codes=True)

                        code_list.append(edge_code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes)

                # Compute logits

                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]
                    edge_label_index = batch.edge_label_index
                    y = batch.edge_label

                    z = model.encode(x, edge_index, edge_attr)
                    edge_z = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2

                    if use_proto_clf:
                        if query_node_code_first:
                            # Case 1: Use node code to form edge code
                            code, commit_loss = model.get_codes(z, use_orig_codes=True)
                            edge_code = (code[edge_label_index[0]] + code[edge_label_index[1]]) / 2
                        else:
                            # Case 2: Query edge code using edge embeddings directly
                            edge_code, commit_loss = model.get_codes(edge_z, use_orig_codes=True)

                        query_emb = edge_z if model.use_z_in_predict else edge_code

                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)

                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                    pred_list.append(pred.detach())
                    y_list.append(y)

                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

                value = evaluate(pred, y, params=params)
                test_values.append(value)

            return {
                'train': np.mean(train_values),
                'val': np.mean(val_values),
                'test': np.mean(test_values),
                'metric': task2metric[params['task']]
            }
