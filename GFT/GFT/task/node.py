import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from utils.others import get_device_from_model, sample_proto_instances, mask2idx
from utils.eval import evaluate, task2metric


def run_node(model, batch_data, batch_labels, batch_ids, optimizer, params, is_train, proto_emb):
    device = get_device_from_model(model)

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    proto_loss = torch.tensor(0.0).to(device)
    act_loss = torch.tensor(0.0).to(device)

    node_feat = getattr(batch_data, 'x', None)
    edge_index = getattr(batch_data, 'edge_index', None)
    edge_attr = getattr(batch_data, 'edge_attr', None) #others.complete_data
    
    z = model.encode(node_feat, edge_index, edge_attr)

    if is_train:
        support_z = z[batch_ids]
        support_code = code[batch_ids]
        support_y = batch_labels.to(device)

        if use_proto_clf:
            code, _ = model.get_codes(z, use_orig_codes=True)
            proto_emb = model.get_class_prototypes(support_code, support_y, params["num_classes"]).detach()
            query_emb = support_z if params['use_z_in_predict'] else code
            proto_loss = model.compute_proto_loss(query_emb, proto_emb, batch_labels) * params["lambda_proto"]
        if use_lin_clf:
            act_loss = model.compute_activation_loss(support_z, support_y) * params["lambda_act"]

        loss = proto_loss + act_loss
        return {
            'proto_loss': proto_loss.item(),
            'act_loss': act_loss.item(),
            'loss': loss.item(),
        }, proto_emb
    
    else:
        if use_proto_clf:
            query_emb = z if model.use_z_in_predict else code
            pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
        if use_lin_clf:
            pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
        
        pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin
        return pred


def eval_node(model, dataset, loader, split, labels, params, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]
    num_classes = params["num_classes"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    pred_proto = 0
    pred_lin = 0

    mini_batch = loader is not None
    if not mini_batch:
        # Encode
        x = dataset.node_text_feat
        edge_index = dataset.edge_index
        edge_attr = dataset.edge_text_feat[dataset.xe]
        y = labels.to(x.device)

        z = model.encode(x, edge_index, edge_attr)

        if setting == "standard":

            if use_proto_clf:
                # Compute Prototypes
                train_mask = split["train"]
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_train, y_train = code[train_mask], y[train_mask]

                proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach()
                query_emb = z if model.use_z_in_predict else code

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            # Evaluate
            train_mask, val_mask, test_mask = split["train"], split["valid"], split["test"]
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

            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                if use_proto_clf:
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_support, y_support = code[s_mask], y[s_mask]
                    z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()
                    query_emb = z_query if params['use_z_in_predict'] else code_query

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate
                value = evaluate(pred, y_query, params=params)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                if use_proto_clf:
                    # Compute Prototypes
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_support, y_support = code[s_mask], y[s_mask]
                    z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()

                    query_emb = z_query if model.use_z_in_predict else code_query

                    # Compute logits
                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)

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
        # The standard setting and the remaining settings are handled differently
        if setting == "standard":
            if use_proto_clf:
                # Define Prototype Loader
                # Prototype instance sampling only for standard setting
                proto_idx = sample_proto_instances(labels, mask2idx(split["train"]),
                                                   num_instances_per_class=model.num_instances_per_class, )
                proto_loader = NeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    input_nodes=proto_idx,
                    batch_size=256,
                    num_workers=8,
                )

                # Encode Prototypes
                code_list, y_list = [], []
                for batch in proto_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_list.append(code.detach())
                    y_list.append(y)

                code = torch.cat(code_list, dim=0)
                y = torch.cat(y_list, dim=0)
                proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

            # Do Prediction
            pred_list, y_list = [], []
            for batch in loader:
                batch = batch.to(device)
                bs = batch.batch_size

                # Encode
                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]

                y = batch.y[:bs]
                z = model.encode(x, edge_index, edge_attr)[:bs]

                if use_proto_clf:
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    query_emb = z if model.use_z_in_predict else code
                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                pred_list.append(pred.detach())
                y_list.append(y)

            pred = torch.cat(pred_list, dim=0)
            y = torch.cat(y_list, dim=0)

            train_mask, val_mask, test_mask = split["train"], split["valid"], split["test"]
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

            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                if use_proto_clf:
                    # Define Loaders for Support and Query Sets
                    # Prototype loader for support set
                    # Query loader for query set
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=kwargs["num_neighbors"],
                        input_nodes=mask2idx(s_mask),
                        batch_size=256,
                        num_workers=8,
                    )
                    query_loader = NeighborLoader(
                        dataset,
                        num_neighbors=kwargs["num_neighbors"],
                        input_nodes=mask2idx(q_mask),
                        batch_size=256,
                        num_workers=8,
                    )

                    # Construct Prototypes based on Support Set
                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)
                        bs = batch.batch_size

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]

                        y = batch.y[:bs]
                        z = model.encode(x, edge_index, edge_attr)[:bs]

                        code, _ = model.get_codes(z, use_orig_codes=True)
                        code_list.append(code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

                # Compute logits
                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    if use_proto_clf:
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        query_emb = z if model.use_z_in_predict else code
                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

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

                if use_proto_clf:
                    # Define Loaders for Support and Query Sets
                    # Prototype loader for support set
                    # Query loader for query set
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=kwargs["num_neighbors"],
                        input_nodes=mask2idx(s_mask),
                        batch_size=256,
                        num_workers=8,
                    )
                    query_loader = NeighborLoader(
                        dataset,
                        num_neighbors=kwargs["num_neighbors"],
                        input_nodes=mask2idx(q_mask),
                        batch_size=256,
                        num_workers=8,
                    )

                    # Construct Prototypes based on Support Set
                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)
                        bs = batch.batch_size

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]

                        y = batch.y[:bs]
                        z = model.encode(x, edge_index, edge_attr)[:bs]

                        code, _ = model.get_codes(z, use_orig_codes=True)
                        code_list.append(code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

                # Compute logits
                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    if use_proto_clf:
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        query_emb = z if model.use_z_in_predict else code
                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

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