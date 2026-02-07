import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model, sample_proto_instances_for_graph


# This works for standard, zero-shot, in-context
def ft_graph(model, dataset, loader, optimizer, split, labels, params, scheduler=None, **kwargs):
    model.train()

    device = get_device_from_model(model)
    setting = params["setting"]
    num_classes = params["num_classes"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    proto_loss = torch.tensor(0.0).to(device)
    act_loss = torch.tensor(0.0).to(device)

    # For few-shot graph-level task, the num_instances_per_class is n_train
    num_instances_per_class = params['num_instances_per_class']

    if use_proto_clf:
        # Define prototype loader

        # Sample Prototypes from each task to form the prototype set
        if setting in ['standard', 'few_shot']:
            # Standard setting contains too much instances
            # Thus we need to do sampling.

            # proto_idx and proto_labels are dictionaries
            proto_idx, proto_labels = sample_proto_instances_for_graph(
                labels, split['train'], num_instances_per_class=num_instances_per_class)
            flat_proto_idx = [item for sublist in proto_idx.values() for item in sublist]
            proto_dataset = dataset[flat_proto_idx]
            proto_loader = DataLoader(proto_dataset, batch_size=1024, num_workers=8)
        else:
            raise NotImplementedError("The setting is not supported for sampling prototype instances.")

        # Encode prototypes
        code_list = []
        for batch in proto_loader:
            batch = batch.to(device)

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat

            # Use graph embedding to query code

            z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
            code, _ = model.get_codes(z, use_orig_codes=True)
            code_list.append(code.detach())

        code = torch.cat(code_list, dim=0)
        proto_emb = model.get_class_prototypes(code, proto_labels, num_classes)

    # Train

    total_proto_loss = 0
    total_act_loss = 0
    total_loss = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat
        y = batch.y.to(torch.float64)

        z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")

        if use_proto_clf:
            code, commit_loss = model.get_codes(z, use_orig_codes=True)
            query_emb = z if params['use_z_in_predict'] else code
            proto_loss = model.compute_proto_loss(query_emb, proto_emb, y, task="multi") * params["lambda_proto"]

        if use_lin_clf:
            act_loss = model.compute_activation_loss(z, y, task="multi") * params["lambda_act"]

        loss = proto_loss + act_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_proto_loss += proto_loss.item()
        total_act_loss += act_loss.item()
        total_loss += loss.item()

    return {
        'proto_loss': total_proto_loss / len(loader),
        'act_loss': total_act_loss / len(loader),
        'loss': total_loss / len(loader),
    }


def eval_graph(model, dataset, loader, split, labels, params, **kwargs):
    train_loader, val_loader, test_loader = loader
    setting = params["setting"]

    if setting == 'standard':

        # train_value = eval_graph_single(model=model, dataset=dataset, loader=train_loader, split=split, labels=labels,
        #                                 params=params, setting=setting)
        train_value = 0

        val_value = eval_graph_single(model=model, dataset=dataset, loader=val_loader, split=split, labels=labels,
                                      params=params, setting=setting)

        test_value = eval_graph_single(model=model, dataset=dataset, loader=test_loader, split=split, labels=labels,
                                       params=params, setting=setting)

        return {
            'train': train_value,
            'val': val_value,
            'test': test_value,
            'metric': task2metric[params["task"]],
        }

    else:
        return eval_graph_few_shot(model=model, dataset=dataset, loader=loader, split=split, labels=labels,
                                   params=params, setting=setting)


# This works for standard setting
def eval_graph_single(model, dataset, loader, split, labels, params, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    num_classes = params["num_classes"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    pred_proto = 0
    pred_lin = 0

    if use_proto_clf:
        # Standard setting contains too much instances
        # Thus we need to do sampling.

        # proto_idx and proto_labels are dictionaries
        proto_idx, proto_labels = sample_proto_instances_for_graph(labels, split['train'],
                                                                   num_instances_per_class=model.num_instances_per_class)
        flat_proto_idx = [item for sublist in proto_idx.values() for item in sublist]
        proto_dataset = dataset[flat_proto_idx]
        proto_loader = DataLoader(proto_dataset, batch_size=1024, num_workers=8)

        code_list = []
        for batch in proto_loader:
            batch = batch.to(device)

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat

            z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
            code, _ = model.get_codes(z, use_orig_codes=True)
            code_list.append(code.detach())

        code = torch.cat(code_list, dim=0)
        proto_emb = model.get_class_prototypes(code, proto_labels, num_classes)

    y_list, pred_list = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat

        z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")

        if use_proto_clf:
            code, commit_loss = model.get_codes(z, use_orig_codes=True)
            query_emb = z if model.use_z_in_predict else code
            pred_proto = model.get_proto_logits(query_emb, proto_emb, task="multi")
        if use_lin_clf:
            pred_lin = model.get_lin_logits(z).mean(1)
        pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

        y_list.append(batch.y.view(pred.shape))
        pred_list.append(pred.detach())

    # Evaluate
    y = torch.cat(y_list, dim=0)
    pred = torch.cat(pred_list, dim=0)
    value = evaluate(pred, y, params=params)

    return value


def eval_graph_few_shot(model, dataset, loader, split, labels, params, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]
    num_classes = params["num_classes"]

    assert setting in ["few_shot"]

    use_proto_clf = not params['no_proto_clf']
    use_lin_clf = not params['no_lin_clf']
    pred_proto = 0
    pred_lin = 0

    n_task = len(split['test']['support']['idx'])
    train_values, val_values, test_values = [], [], []

    # Validation: few-shot, zero-shot, and in-context
    for i in range(n_task):
        s_idx, s_label = split['valid']['support']['idx'][i], split['valid']['support']['label'][i]
        q_idx, q_label = split['valid']['query']['idx'][i], split['valid']['query']['label'][i]

        if use_proto_clf:
            # Get prototypes
            flat_s_idx = [item for sublist in s_idx.values() for item in sublist]
            proto_dataset = dataset[flat_s_idx]
            proto_loader = DataLoader(
                proto_dataset,
                batch_size=1024,
                num_workers=8,
            )

            code_list = []
            for batch in proto_loader:
                batch = batch.to(device)

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat

                z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_list.append(code.detach())

            code = torch.cat(code_list, dim=0)
            proto_emb = model.get_class_prototypes(code, s_label, num_classes).detach()

        # Prediction

        flat_q_idx = [item for sublist in q_idx.values() for item in sublist]
        query_dataset = dataset[flat_q_idx]
        query_loader = DataLoader(
            query_dataset,
            batch_size=1024,
            num_workers=8,
        )

        y_list, pred_list = [], []
        for step, batch in enumerate(query_loader):
            batch = batch.to(device)

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat

            z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
            if use_proto_clf:
                code, commit_loss = model.get_codes(z, use_orig_codes=True)
                query_emb = z if model.use_z_in_predict else code
                pred_proto = model.get_proto_logits(query_emb, proto_emb, task="multi")
            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1)
            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            y_list.append(batch.y.view(pred.shape))
            pred_list.append(pred.detach())

        # Evaluate
        y = torch.cat(y_list, dim=0)
        pred = torch.cat(pred_list, dim=0)
        value = evaluate(pred, y, params=params)

        train_values.append(value)
        val_values.append(value)

    for i in range(n_task):
        s_idx, s_label = split['test']['support']['idx'][i], split['test']['support']['label'][i]
        q_idx, q_label = split['test']['query']['idx'][i], split['test']['query']['label'][i]

        if use_proto_clf:
            # Get prototypes

            flat_s_idx = [item for sublist in s_idx.values() for item in sublist]
            proto_dataset = dataset[flat_s_idx]
            proto_loader = DataLoader(
                proto_dataset,
                batch_size=1024,
                num_workers=8,
            )

            code_list = []
            for batch in proto_loader:
                batch = batch.to(device)

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat

                z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_list.append(code.detach())

            code = torch.cat(code_list, dim=0)
            proto_emb = model.get_class_prototypes(code, s_label, num_classes).detach()

        # Prediction

        flat_q_idx = [item for sublist in q_idx.values() for item in sublist]
        query_dataset = dataset[flat_q_idx]
        query_loader = DataLoader(
            query_dataset,
            batch_size=1024,
            num_workers=8,
        )

        y_list, pred_list = [], []
        for step, batch in enumerate(query_loader):
            batch = batch.to(device)

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat

            z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")

            if use_proto_clf:
                code, commit_loss = model.get_codes(z, use_orig_codes=True)
                query_emb = z if model.use_z_in_predict else code
                pred_proto = model.get_proto_logits(query_emb, proto_emb, task="multi")
            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1)
            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            y_list.append(batch.y.view(pred.shape))
            pred_list.append(pred.detach())

        # Evaluate
        y = torch.cat(y_list, dim=0)
        pred = torch.cat(pred_list, dim=0)
        value = evaluate(pred, y, params=params)

        test_values.append(value)

    return {
        'train': np.mean(train_values),
        'val': np.mean(val_values),
        'test': np.mean(test_values),
        'metric': task2metric[params["task"]],
    }