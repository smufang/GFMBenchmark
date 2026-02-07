from exp.exp_basic import ExpBasic
import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import is_undirected
from data_provider.subbatch_loader import BatchGraphLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score
from utils.tools import EarlyStopping
import time
from tqdm import tqdm
import copy


class ExpDownstreamBatch(ExpBasic):
    def __init__(self, args, pretrain_dict, name, dataset):
        args.target_data = name
        super(ExpDownstreamBatch, self).__init__(args, pretrain_dict)
        self._print_main(f"Target dataset: {name}")

        self.need_y = False

        self.dataset = dataset()
        preprocess_map = {
            "basic": self._basic_preprocess,
            "simple": self._simple_preprocess,
        }
        try:
            self.data = preprocess_map[args.preprocess](self.dataset)
        except KeyError:
            raise ValueError(f"Unknown preprocess mode: {args.preprocess}")

        if self.compress_func is not None:
            self.data = self._pure_data(self._get_compressed_data(self.data))
        else:
            self.data.x = self.data.x.float()

        self.task_name = self.args.task_name
        self.is_batch = True if self.task_name == "graph" else False 
        self.only_test_batch = True if self.task_name == "graph" else False

        if self.args.criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.args.criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.args.criterion == 'nll':
            self.criterion = nn.NLLLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown criterion: {self.args.criterion}")
        
        if args.preprocess == 'basic':
            self.avg_degree = self.data.num_edges / self.data.num_nodes
            self._print_main(f"Average degree: {self.avg_degree:.2f}")
            self._write_log(f"Average degree: {self.avg_degree:.2f}")
            self.is_undirected = is_undirected(self.data.edge_index)
            self._print_main(f"Is undirected: {self.is_undirected}")
            self._write_log(f"Is undirected: {self.is_undirected}")

    def _get_pretrain_model(self, pretrain_model):
        self.pretrain_setting = self._simple_pretrain_setting()
        path = (
                self.args.checkpoints + "/" + self.pretrain_setting + "/" + "checkpoint.pth"
        )

        self._load_checkpoint(path, pretrain_model)
        return pretrain_model
    
    def _get_data(self):
        """Get original dataset without compression"""
        from torch_geometric.data import Data, HeteroData, TemporalData
        from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
        from data_provider.data_loader import create_x, complete_data

        if len(self.dataset) > 1:
            data = multi_to_one(self.dataset, need_y=self.need_y)
            self.data_type = "multi"
        else:
            data = self.dataset[0]
            if isinstance(data, TemporalData):
                data = temporal_to_data(data, need_y=self.need_y)
                self.data_type = "temporal"
            elif isinstance(data, HeteroData):
                data = hetero_to_data(data, need_y=self.need_y)
                self.data_type = "hetero"
            elif isinstance(data, Data):
                data = create_x(data)
                self.data_type = "data"
            else:
                raise ValueError(f"Unknown data type: {type(data)}")
        data = complete_data(data, self.args.target_data, need_y=self.need_y)
        return data

    def _get_few_shot(self):
        from data_provider.fewshot_generator import load_few_shot_tasks

        return load_few_shot_tasks(
            self.args.target_data, task=self.task_name, K=self.args.num_shots, num_tasks=self.args.num_tasks
        )

    def _get_test(self):
        from data_provider.fewshot_generator import load_test_splits

        return load_test_splits(self.args.target_data, task=self.task_name)
    
    def _select_optimizer(self, param):
        model_optim = optim.Adam(
            param, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        
        scheduler = lr_scheduler.MultiStepLR(
            model_optim, milestones=[400], gamma=0.1
        )
        
        return model_optim, scheduler

    def _create_progress_bar(self, few_shot_tasks, task_dim):
        if self._is_main_process():
            return tqdm(
                few_shot_tasks,
                desc=f"Task Dim {task_dim}",
                position=0,
                leave=True,
                dynamic_ncols=False,
            )
        else:
            return few_shot_tasks

    def _get_loader(self, data, task_labels, task_ids, is_train):
        batch_size = self.args.batch_size
        persistent_workers=False

        if self.task_name == "node":
            if self.data_type == 'hetero':
                num_neighbors = [5, 5, 5, 5]
                directed = False
            else:
                num_neighbors = [10, 10, 10, 10] #[5, 5, 5, 5]
                directed = True
            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                input_nodes=task_ids,
                batch_size=batch_size,
                shuffle=False,
                directed=directed,
                subgraph_type='bidirectional',
                replace=False,
                num_workers=self.args.num_workers,
                persistent_workers=persistent_workers
            )
            for batch in loader:
                batch_labels = task_labels[batch.input_id]
                batch_ids = torch.arange(len(batch_labels), dtype=torch.long)
                yield batch, batch_labels, batch_ids

        elif self.task_name == "edge":
            loader = LinkNeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5],
                edge_label_index=data.edge_index[:, task_ids],
                edge_label=task_labels,
                batch_size=batch_size,
                shuffle=False,
                directed=False,
                subgraph_type='directional',
                replace=False,
                num_workers=self.args.num_workers,
                persistent_workers=persistent_workers
            )
            for batch in loader:
                yield batch, batch.edge_label, batch.edge_label_index

        elif self.task_name == "graph":
            loader = BatchGraphLoader(
                data,
                graph_label_index=task_ids,
                graph_label=task_labels,
                batch_size=batch_size, # recommend 8192 * 4
                shuffle=False,
                num_workers=self.args.num_workers
            )
            for batch in loader:
                yield batch, batch.graph_label, batch.graph_label_index

    def run_model_cache(self, task, is_train, cache=None):
        # only for graph level
        data = Data(x=self.data.x, edge_index=self.data.edge_index, batch=self.data.batch)
        task_labels = torch.tensor(task["labels"], dtype=torch.int64)
        task_ids = torch.tensor(task["idx"], dtype=torch.int64)

        if cache is None:
            if (is_train is False and self.only_test_batch is True) or self.is_batch is True:
                loader = self._get_loader(data, task_labels, task_ids, is_train)
                cache = list(loader)
            else:
                if self.task_name == 'edge':
                    task_ids = data.edge_index[:, task_ids]
                cache = [(data, task_labels, task_ids)]
        
        labels = []
        total_loss = torch.zeros(1, device=self.device, requires_grad=False)
        preds = []
        for batch, batch_labels, batch_ids in cache:
            if is_train is True:
                self.optimizer.zero_grad()
            try:
                batch = batch.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_ids = batch_ids.to(self.device)
                batch.name = [self.args.target_data] * batch.num_nodes
                batch_logits = self.model(batch, batch_labels, batch_ids, is_train=is_train)
            except Exception as e:
                if not self.is_batch and not self.only_test_batch:
                    if is_train is True:
                        self._print_main(f"{e}\n Switching to batch mode.")
                        self._write_log("Switching to batch mode.")
                        self.is_batch = True
                        return self.run_model_cache(task, is_train, cache=cache)
                    else:
                        self._print_main(f"{e}\n Testing to batch mode.")
                        self._write_log("Testing to batch mode.")
                        self.only_test_batch = True
                        return self.run_model_cache(task, is_train, cache=cache)
                else:
                    self._print_main(f"Num Nodes: {batch.num_nodes}")
                    self._print_main(f"Num Edges: {batch.num_edges}")
                    raise e
                
            if is_train is True:
                loss = self.criterion(batch_logits, batch_labels)
                total_loss += loss.item() * batch_labels.size(0)
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                # test on cpu
                labels.append(batch_labels.cpu())
                pred = torch.argmax(batch_logits, dim=1).cpu()
                preds.append(pred)

            del batch, batch_labels, batch_ids, batch_logits
            torch.cuda.empty_cache()

        if is_train is True:
            loss = total_loss / task_labels.size(0)
            return loss, cache
        else:
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            return preds, labels, cache


    def run_model(self, task, is_train):
        data = Data(x=self.data.x, edge_index=self.data.edge_index, batch=self.data.batch)
        task_labels = torch.tensor(task["labels"], dtype=torch.int64)
        task_ids = torch.tensor(task["idx"], dtype=torch.int64)

        if (is_train is False and self.only_test_batch is True) or self.is_batch is True:
            loader = self._get_loader(data, task_labels, task_ids, is_train)
        else:
            if self.task_name == 'edge':
                task_ids = data.edge_index[:, task_ids]
            loader = [(data, task_labels, task_ids)]
        labels = []
        total_loss = torch.zeros(1, device=self.device, requires_grad=False)
        preds = []
        for batch, batch_labels, batch_ids in loader:
            if is_train is True:
                self.optimizer.zero_grad()
            try:
                batch = batch.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_ids = batch_ids.to(self.device)
                batch.name = [self.args.target_data] * batch.num_nodes
                batch_logits = self.model(batch, batch_labels, batch_ids, is_train=is_train)
            except Exception as e:
                if not self.is_batch and not self.only_test_batch:
                    if is_train is True:
                        self._print_main(f"{e}\n Switching to batch mode.")
                        self._write_log("Switching to batch mode.")
                        self.is_batch = True
                        return self.run_model(task, is_train)
                    else:
                        self._print_main(f"{e}\n Testing to batch mode.")
                        self._write_log("Testing to batch mode.")
                        self.only_test_batch = True
                        return self.run_model(task, is_train)
                else:
                    self._print_main(f"Num Nodes: {batch.num_nodes}")
                    self._print_main(f"Num Edges: {batch.num_edges}")
                    raise e
                
            if is_train is True:
                loss = self.criterion(batch_logits, batch_labels)
                total_loss += loss.item() * batch_labels.size(0)
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                # test on cpu
                labels.append(batch_labels.cpu())
                pred = torch.argmax(batch_logits, dim=1).cpu()
                preds.append(pred)

            del batch, batch_labels, batch_ids, batch_logits
            torch.cuda.empty_cache()

        if is_train is True:
            loss = total_loss / task_labels.size(0)
            return loss
        else:
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            return preds, labels

    def run_tasks(self, model, param):
        ini_model_state = copy.deepcopy(model.state_dict()) # save initial model state
        self.model = model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda") if self.args.use_amp else None
        fewshot_tasks_dict = self._get_few_shot()
        test_split_dict = self._get_test()

        all_acc_list = []
        all_microf_list = []
        all_macrof_list = []

        for task_dim, fewshot_tasks in fewshot_tasks_dict.items():
            time_start = time.time()
            test_task = test_split_dict[task_dim]
            self.args.num_classes = test_task["num_classes"]

            acc_lst = []
            macrof_lst = []
            microf_lst = []

            self._print_main(
                f"{self.task_name.capitalize()} Classification on {self.args.target_data} "
                f"Y-Dim({task_dim}) with {self.args.num_classes}-class {self.args.num_shots}-shot"
            )
            self._write_log(
                f"{self.task_name.capitalize()} Classification on {self.args.target_data} "
                f"Y-Dim({task_dim}) with {self.args.num_classes}-class {self.args.num_shots}-shot"
            )
            test_cache = None
            pbar = self._create_progress_bar(fewshot_tasks, task_dim)
            total_epochs = 0
            for idx, train_task in enumerate(pbar, start=1):
                self._monitor_resources()
                self.model.load_state_dict(ini_model_state) # reset model
                self.optimizer, scheduler = self._select_optimizer(param)
                self.model.train()
                early_stopping = EarlyStopping(
                    patience=self.args.patience, verbose=True
                )
                task_time_start = time.time()
                train_cache = None
                for epoch in range(self.args.epochs):
                    if self.task_name == "graph":
                        train_loss, train_cache = self.run_model_cache(train_task, is_train=True, cache=train_cache)
                    else:
                        train_loss = self.run_model(train_task, is_train=True)
                    
                    scheduler.step()
                    
                    self._print_main(
                        f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {train_loss.item():.4f}"
                    )

                    early_stopping(train_loss.item(), self.model, path=None, is_save=False)
                    if early_stopping.early_stop:
                        self._print_main(f"Early stopping at epoch: {epoch + 1}")
                        self._write_log(f"Early stopping at epoch: {epoch + 1}")
                        break
                
                self._monitor_resources(gpu_threshold=20)
                early_stopping.load_best_model(self.model, device=self.device)
                self.model.eval()
                with torch.no_grad():
                    if self.task_name == "graph":
                        preds, test_labels, test_cache = self.run_model_cache(test_task, is_train=False, cache=test_cache)
                    else:
                        preds, test_labels = self.run_model(test_task, is_train=False)

                acc = torch.sum(preds == test_labels).float() / test_labels.size(0)
                preds_cpu = preds.cpu().numpy()
                test_lbls_cpu = test_labels.cpu().numpy()
                micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="micro")
                macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="macro")
                microf_lst.append(micro_f1 * 100)
                macrof_lst.append(macro_f1 * 100)
                acc_lst.append(acc * 100)
                task_time_end = time.time()
                if self._is_main_process() and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        {
                            "Test Acc": f"{acc * 100:.2f}%",
                            "Micro-F1": f"{micro_f1 * 100:.2f}%",
                            "Macro-F1": f"{macro_f1 * 100:.2f}%",
                        }
                    )
                self._write_log(
                    f"Task {idx} -- Test Acc: {acc * 100:.2f}%, Micro-F1: {micro_f1 * 100:.2f}%, "
                    f"Macro-F1: {macro_f1 * 100:.2f}%, Time: {task_time_end - task_time_start:.2f}s"
                )
                total_epochs += epoch + 1

            acc_tensor = torch.stack(acc_lst)
            acc_mean = acc_tensor.mean().item()
            microf_mean = sum(microf_lst) / len(microf_lst)
            macrof_mean = sum(macrof_lst) / len(macrof_lst)
            acc_std = acc_tensor.std().item()
            microf_std = torch.std(torch.tensor(microf_lst)).item()
            macrof_std = torch.std(torch.tensor(macrof_lst)).item()
            self._print_main(
                f'===Y-Dim({task_dim}){"=" * 50}\n'
                f"Accuracy:[{acc_mean:.4f}±{acc_std:.4f}]\n"
                f"Micro-F1:[{microf_mean:.4f}±{microf_std:.4f}]\n"
                f"Macro-F1:[{macrof_mean:.4f}±{macrof_std:.4f}]\n"
                f"Total Time: {time.time() - time_start:.2f}s\n"
                f"Total Epochs: {total_epochs}\n"
                f"Average Time per epoch: {(time.time() - time_start)/total_epochs:.4f}s\n"
            )
            self._write_log(
                f'===Y-Dim({task_dim}){"=" * 50}\n'
                f"Accuracy:[{acc_mean:.4f}±{acc_std:.4f}]\n"
                f"Micro-F1:[{microf_mean:.4f}±{microf_std:.4f}]\n"
                f"Macro-F1:[{macrof_mean:.4f}±{macrof_std:.4f}]\n"
                f"Total Time: {time.time() - time_start:.2f}s\n"
                f"Total Epochs: {total_epochs}\n"
                f"Average Time per epoch: {(time.time() - time_start)/total_epochs:.4f}s\n"
            )
            all_acc_list.extend(acc_lst)
            all_microf_list.extend(microf_lst)
            all_macrof_list.extend(macrof_lst)

        if len(fewshot_tasks_dict) > 1:
            all_acc_tensor = torch.stack(all_acc_list)
            all_acc_mean = all_acc_tensor.mean().item()
            all_microf_mean = sum(all_microf_list) / len(all_microf_list)
            all_macrof_mean = sum(all_macrof_list) / len(all_macrof_list)
            all_acc_std = all_acc_tensor.std().item()
            all_microf_std = torch.std(torch.tensor(all_microf_list)).item()
            all_macrof_std = torch.std(torch.tensor(all_macrof_list)).item()
            self._write_log(
                f'===Overall{"=" * 50}\n'
                f"Accuracy:[{all_acc_mean:.4f}±{all_acc_std:.4f}]\n"
                f"Micro-F1:[{all_microf_mean:.4f}±{all_microf_std:.4f}]\n"
                f"Macro-F1:[{all_macrof_mean:.4f}±{all_macrof_std:.4f}]\n"
            )
