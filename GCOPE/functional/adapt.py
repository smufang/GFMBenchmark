import copy
from fastargs.decorators import param
import torch
from torch_geometric.data import Data
from data_provider import *
from exp.exp_downstream_simple import ExpDownstreamBatch
from datetime import datetime
import time
import numpy as np

class Args:
    def __init__(self, task_name, model_id, exp_id, num_shots=1, batch_size=64, num_tasks=50,seed=0):
        self.model = 'gcope'
        self.model_id = model_id
        self.exp_id = exp_id
        self.task_name = task_name
        self.pattern = 'simple'
        self.preprocess = 'basic'
        self.mode = 'gcl'
        self.backbone = 'fagcn'
        self.compress_function = 'svd_gcope'
        self.combinetype = 'none'
        self.input_dim = 100
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_heads = 0
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.seed = seed
        self.is_logging = True
        self.num_workers = 4
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


exps_downstream = {
    f"exp{i}": {
        "node": globals()[f"NC_exp{i}"],
        "edge": globals()[f"EC_exp{i}"],
        "graph": globals()[f"GC_exp{i}"],
    }
    for i in range(5)
}


@param('general.model_id', 'model_id')
@param('general.exp_id', 'exp_id')
@param('general.seed', 'seed')
@param('data.name', 'dataset')
@param('general.few_shot', 'few_shot')
@param('adapt.batch_size', 'batch_size')
@param('adapt.repeat_times')
def run(
    model_id,
    exp_id,
    seed,
    dataset,
    few_shot,
    batch_size,
    repeat_times,
    ):
    task_name = dataset[0]
    args = Args(model_id=model_id, exp_id=exp_id, task_name=task_name, num_shots=few_shot, batch_size=batch_size, num_tasks=repeat_times, seed=seed)
    dataset_dict = exps_downstream[args.exp_id][dataset[0]]

    for name, generator in dataset_dict.items():
        print(f'Adapting on {name} dataset...')
        run_one(exp = ExpDownstreamBatch(args, pretrain_dict=None, name=name, dataset=generator))

# @param('data.name', 'dataset')
# @param('adapt.batch_size')
# @param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
def run_one(
    exp,
    method,
    backbone_model,
    saliency_model,
    answering_model,
    pretrained_file,
    save_dir,
    ):
    # origin_data = exp.data
    # data = Data()
    # data.x = origin_data.x.contiguous()
    # data.edge_index = origin_data.edge_index.contiguous()
    # if hasattr(origin_data, "batch") and exp.args.task_name == "graph-simple":
    #     data.batch = origin_data.batch  # Graph-level
    fewshot_tasks_dict = exp._get_few_shot()
    test_split_dict = exp._get_test()
    total_results = []
    for task_dim, fewshot_tasks in fewshot_tasks_dict.items():
        time_start = time.time()
        test_task = test_split_dict[task_dim]
        num_classes = test_task["num_classes"]

        exp._print_main(
                f"{exp.args.task_name.capitalize()} Classification on {exp.args.target_data} "
                f"Y-Dim({task_dim}) with {num_classes}-class {exp.args.num_shots}-shot"
            )
        exp._write_log(
            f"{exp.args.task_name.capitalize()} Classification on {exp.args.target_data} "
            f"Y-Dim({task_dim}) with {num_classes}-class {exp.args.num_shots}-shot"
        )

        # init model
        from GCOPE.model import get_model
        model_ini = get_model(
            backbone_kwargs = {
                'name': backbone_model,
                'num_features': exp.args.input_dim,
            },
            answering_kwargs = {
                'name': answering_model,
                'hid_dim': exp.args.hidden_dim * 2 if exp.args.task_name.split('-')[0] == 'edge' else exp.args.hidden_dim,
                'num_class': num_classes,
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_dim': exp.args.input_dim,
            } if saliency_model != 'none' else None,
        )

        model_ini.load_state_dict(torch.load(pretrained_file), strict=False)
        exp._print_main(f'Loaded pretrained model from {pretrained_file}')
        # train
        all_results = []
        for idx, train_task in enumerate(fewshot_tasks, start=1):
            # train_labels = torch.tensor(train_task['labels'])
            # train_ids = torch.tensor(train_task['idx'])
            # loaders = {
            #     'train': exp._get_loader_subgraph(data, train_labels, train_ids, is_train=True),
            #     'test': exp._get_loader_subgraph(data, test_labels, test_ids, is_train=False),
            # }
            model = copy.deepcopy(model_ini)
            exp._monitor_resources()
            task_time_start = time.time()
            if method == 'finetune':
                results = finetune(exp, model, train_task, test_task)
            # elif method == 'prog':
            #     from GCOPE.model import get_prompt_model
            #     # statistic the average node number of dataset
            #     total_graph = sum([len(v) for k, v in datasets.items()])
            #     train_node_num = sum([g.num_nodes for g in datasets['train']])
            #     # val_node_num = sum([g.num_nodes for g in datasets['val']])
            #     test_node_num = sum([g.num_nodes for g in datasets['test']])
            #     prompt_node_num = int((train_node_num + test_node_num) / total_graph)
            #     prompt_model = get_prompt_model(num_features= exp.args.input_dim, prompt_node_num=prompt_node_num)
            #     results = prog(loaders, model, prompt_model, dataset)        
            else:
                raise NotImplementedError(f'Unknown method: {method}')
            
            # results.pop('model')
            task_time_end = time.time()
            all_results.append(results)
            total_results.append(results)

            exp._print_main(
                f"Task {idx} -- Test Acc: {results['acc'] * 100:.2f}%, AUC: {results['auroc'] * 100:.2f}%, "
                f"Macro-F1: {results['f1'] * 100:.2f}%, Time: {task_time_end - task_time_start:.2f}s"
            )
            exp._write_log(
                f"Task {idx} -- Test Acc: {results['acc'] * 100:.2f}%, AUC: {results['auroc'] * 100:.2f}%, "
                f"Macro-F1: {results['f1'] * 100:.2f}%, Time: {task_time_end - task_time_start:.2f}s"
                )
        # print acc, auroc, f1 with std
        metric_strs = []
        for k in all_results[0].keys():
            mean_val = np.mean([r[k] for r in all_results]) * 100
            std_val = np.std([r[k] for r in all_results]) * 100
            metric_strs.append(f"{k.capitalize()}:[{mean_val:.4f}±{std_val:.4f}]")

        total_epochs = int(exp.args.num_tasks * 100)
        summary = (
            f'===Y-Dim({task_dim}){"=" * 50}\n'
            + "\n".join(metric_strs) + "\n"
            + f"Total Time: {time.time() - time_start:.2f}s\n"
            + f"Total Epochs: {total_epochs}\n"
            + f"Average Time per epoch: {(time.time() - time_start)/total_epochs:.4f}s\n"
        )

        exp._print_main(summary)
        exp._write_log(summary)

    if len(fewshot_tasks_dict) > 1:
        # print overall acc, auroc, f1 with std
        metric_strs = []
        for k in total_results[0].keys():
            mean_val = np.mean([r[k] for r in total_results]) * 100
            std_val = np.std([r[k] for r in total_results]) * 100
            metric_strs.append(f"{k.capitalize()}:[{mean_val:.4f}±{std_val:.4f}]")

        total_epochs = int(len(total_results) * 100)
        summary = (
            f'===Overall{"=" * 50}\n'
            + "\n".join(metric_strs) + "\n"
        )
        exp._print_main(summary)
        exp._write_log(summary)


@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        exp,
        model,
        train_task,
        test_task,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
        ):

    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate,
        weight_decay = weight_decay,
        )
    test_labels = torch.tensor(test_task['labels'],dtype=torch.int64)
    test_ids = torch.tensor(test_task['idx'],dtype=torch.int64)
    train_labels = torch.tensor(train_task['labels'],dtype=torch.int64)
    train_ids = torch.tensor(train_task['idx'],dtype=torch.int64)

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_loss = float('inf')
    best_model = None
    best_epoch = -1
    for e in range(epoch):
        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        train_loader = exp._get_loader_subgraph(exp.data, train_labels, train_ids, is_train=True)
        pbar = tqdm(train_loader, ncols=100, desc=f'Epoch {e} Training')
        total_loss = 0.
        for batch, batch_labels, batch_ids in pbar:
            batch = batch.to(device)
            batch_labels = batch_labels.to(device)
            batch_ids = batch_ids.to(device)
            optimizer.zero_grad()
            pred = model(batch, batch_ids, task_type=exp.args.task_name)
            loss = torch.nn.functional.cross_entropy(pred, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.detach(), batch_labels.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()
        if best_loss > total_loss / train_labels.size(0):
            best_loss = total_loss / train_labels.size(0)
            best_epoch = e
            best_model = deepcopy(model)
    
    model = best_model if best_model is not None else model
    exp._write_log(f'Best epoch: {best_epoch}, Best training loss: {best_loss:.4f}')
    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()
    test_loader = exp._get_loader_subgraph(exp.data, test_labels, test_ids, is_train=False)
    pbar = tqdm(test_loader, ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch, batch_labels, batch_ids in pbar:
            batch = batch.to(device)
            batch_labels = batch_labels.to(device)
            batch_ids = batch_ids.to(device)
            logits = model(batch, batch_ids, task_type=exp.args.task_name)
            pred = logits.argmax(dim=-1)

            acc_metric.update(pred, batch_labels)
            f1_metric.update(pred, batch_labels)
            # 这边计算了两次
            # auroc_metric.update(model(batch, batch_ids, task_type=exp.args.task_name), batch_labels)
            auroc_metric.update(logits, batch_labels)

            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()
    
    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        #'model': model.state_dict(),
    }

@param('adapt.epoch')
@param('adapt.prog.prompt_lr')
@param('adapt.prog.prompt_weight_decay')
@param('adapt.prog.ans_lr')
@param('adapt.prog.ans_weight_decay')
@param('adapt.prog.backbone_tuning')
@param('adapt.prog.saliency_tuning')
def prog(
        loaders,
        model,
        prompt_model,      
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,          
        prompt_lr,
        prompt_weight_decay,
        ans_lr,
        ans_weight_decay,
        ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    prompt_model.to(device)
    
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    opi_pg = torch.optim.Adam(
        prompt_model.parameters(),
        lr = prompt_lr,
        weight_decay = prompt_weight_decay,
        )
    
    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr = ans_lr,
        weight_decay = ans_weight_decay,
        )    

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)
    
    # load prompting data

    from torch_geometric.loader import DataLoader

    best_acc = 0.
    best_backbone = None
    best_prompt_model = None
    best_answering = None

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        
        print(("{}/{} frozen gnn | *tune prompt and tune answering function...".format(e, epoch)))
        prompt_model.train()
        model.backbone.eval()
        model.answering.train()

        from tqdm import tqdm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        running_loss = 0.
        
        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2       
            
            train_batch = train_batch.to(device)
            prompted_graph = prompt_model(train_batch)

            graph_emb = model.backbone(prompted_graph)

            # print(graph_emb)
            pred = model.answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            ans_pbar.set_description('Epoch {} / Total Epoch {} | avg loss: {:.8f}'.format(e, epoch, current_avg_last_loss), refresh=True)
        
        ans_pbar.close()        
                
        model.backbone.eval()
        prompt_model.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:              
                batch = batch.to(device)
                prompted_graph = prompt_model(batch)
                z = model.backbone(prompted_graph)
                pred = model.answering(z).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(prompted_graph), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(prompt_model)
    
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    prompt_model = best_prompt_model if best_prompt_model is not None else prompt_model

    # test
    model.backbone.eval()
    model.answering.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            prompted_graph = prompt_model(batch)
            z = model.backbone(prompted_graph)
            pred = model.answering(z).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(prompted_graph), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        # 'model': model.state_dict(),
    }