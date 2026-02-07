import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from GraphCLIP.model import GraphCLIP
from GraphCLIP.utils.args import Arguments
from GraphCLIP.gen_target_subg import ExpDownstreamBatchGraphCLIP

from layers.Prompt import TextPrompt
from utils.tools import EarlyStopping
import copy


class TuningLayers(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TuningLayers, self).__init__()
        self.prompt = TextPrompt(input_dim, combinetype="add")
        self.edge_linear = nn.Linear(in_features=output_dim, out_features=2*output_dim)

    def forward_prompt(self, x):
        x = self.prompt(x)
        return x

    def forward_edge(self, x):
        x = self.edge_linear(x)
        return x


def prompt_tuning(exp, loader, prompt, num_labels, is_train, device):
    total_loss = torch.zeros(1, device=device)
    labels = []
    preds = []
    batch_t = exp.tokenizer(exp.class_texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)

    for i, (batch, ids, batch_labels) in enumerate(loader):
        batch = batch.to(device)

        batch.x = prompt.forward_prompt(batch.x)
        graph_embs, _ = exp.pretrain_model.encode_graph(batch) # [B, D]
        text_embs = exp.pretrain_model.prompt_text(batch_t["input_ids"], batch_t['token_type_ids'], batch_t["attention_mask"]) # [C, D]

        if exp.task_name == "edge":
            src, dst = ids.to(device)
            graph_embs = torch.concat([graph_embs[src], graph_embs[dst]], dim=-1) # [B, 2*D]
            text_embs = prompt.forward_edge(text_embs) # [C, 2*D]

        graph_embs = graph_embs / graph_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        if is_train:
            batch_labels = batch_labels.to(device)
            text_embs = text_embs[batch_labels] # [C, D] -> [B, D]
            features = torch.stack([graph_embs, text_embs], dim=1) # [B,2,D] two view from graph and text
            total_loss += exp.pretrain_model.sup_loss(features, labels=batch_labels, mask=None) * len(batch_labels)
        else:
            similarity = (100.0 * graph_embs @ text_embs.T).softmax(dim=-1)
            pred = similarity.argmax(dim=-1)
            preds.append(pred.cpu())
            labels.append(batch_labels.cpu())
        
    if is_train:
        return total_loss / num_labels
    else:
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        return preds, labels
    
def main(args, pretrain_dict, name, dataset):
    exp = ExpDownstreamBatchGraphCLIP(args, pretrain_dict, name, dataset)
    attn_kwargs = {'dropout': 0.0}
    model = GraphCLIP(args.input_dim, args.hidden_dim, 12, attn_kwargs, text_model=args.lm_type)
    model.freeze_text()
    model.freeze_graph()
    exp.pretrain_model = exp._get_pretrain_model(model, strict=False)
    ini_model_state = copy.deepcopy(exp.pretrain_model.state_dict()) # save initial model state
    exp.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    fewshot_tasks_dict = exp._get_few_shot()
    test_split_dict = exp._get_test()

    all_acc_list = []
    all_microf_list = []
    all_macrof_list = []

    for task_dim, fewshot_tasks in fewshot_tasks_dict.items():
        time_start = time.time()
        test_task = test_split_dict[task_dim]
        exp.args.num_classes = test_task["num_classes"]

        acc_lst = []
        macrof_lst = []
        microf_lst = []

        exp._print_main(
            f"{exp.task_name.capitalize()} Classification on {exp.args.target_data} "
            f"Y-Dim({task_dim}) with {exp.args.num_classes}-class {exp.args.num_shots}-shot"
        )
        exp._write_log(
            f"{exp.task_name.capitalize()} Classification on {exp.args.target_data} "
            f"Y-Dim({task_dim}) with {exp.args.num_classes}-class {exp.args.num_shots}-shot"
        )
        pbar = exp._create_progress_bar(fewshot_tasks, task_dim)
        total_epochs = 0
        for idx, train_task in enumerate(pbar, start=1):
            exp._monitor_resources()

            prompt = TuningLayers(args.input_dim, args.output_dim).to(args.device)
            prompt.train()
            exp.pretrain_model.load_state_dict(ini_model_state) # reset model to initial state
            exp.pretrain_model.to(args.device)
            exp.pretrain_model.train()

            learnable_params = list(prompt.parameters()) + list(exp.pretrain_model.prefix_encoder.parameters())
            optimizer = torch.optim.AdamW(learnable_params, lr=args.lr, weight_decay=args.weight_decay)

            early_stopping_prompt = EarlyStopping(
                    patience=args.patience, verbose=True
                )
            early_stopping_model = EarlyStopping(
                    patience=args.patience, verbose=False
                )
            task_time_start = time.time()
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                loader = exp._get_loader_subgraph(train_task)
                loss = prompt_tuning(exp, loader, prompt, num_labels=len(train_task["labels"]), is_train=True, device=args.device)
                loss.backward()
                optimizer.step()
                exp._print_main(
                    f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}"
                )

                early_stopping_prompt(loss.item(), prompt, path=None, is_save=False)
                early_stopping_model(loss.item(), exp.pretrain_model, path=None, is_save=False)
                if early_stopping_prompt.early_stop:
                    exp._print_main(f"Early stopping at epoch: {epoch + 1}")
                    exp._write_log(f"Early stopping at epoch: {epoch + 1}")
                    break
            
            early_stopping_prompt.load_best_model(prompt, device=args.device)
            early_stopping_model.load_best_model(exp.pretrain_model, device=args.device)
            prompt.eval()
            exp.pretrain_model.eval()
            with torch.no_grad():
                loader = exp._get_loader_subgraph(test_task)
                preds, test_labels = prompt_tuning(exp, loader, prompt, num_labels=len(test_task["labels"]), is_train=False, device=args.device)
            acc = torch.sum(preds == test_labels).float() / test_labels.size(0)
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_labels.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="micro")
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average="macro")
            microf_lst.append(micro_f1 * 100)
            macrof_lst.append(macro_f1 * 100)
            acc_lst.append(acc * 100)
            task_time_end = time.time()
            if exp._is_main_process() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    {
                        "Test Acc": f"{acc * 100:.2f}%",
                        "Micro-F1": f"{micro_f1 * 100:.2f}%",
                        "Macro-F1": f"{macro_f1 * 100:.2f}%",
                    }
                )
            exp._write_log(
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
        exp._print_main(
            f'===Y-Dim({task_dim}){"=" * 50}\n'
            f"Accuracy:[{acc_mean:.4f}±{acc_std:.4f}]\n"
            f"Micro-F1:[{microf_mean:.4f}±{microf_std:.4f}]\n"
            f"Macro-F1:[{macrof_mean:.4f}±{macrof_std:.4f}]\n"
            f"Total Time: {time.time() - time_start:.2f}s\n"
            f"Total Epochs: {total_epochs}\n"
            f"Average Time per epoch: {(time.time() - time_start)/total_epochs:.4f}s\n"
        )
        exp._write_log(
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
        exp._write_log(
            f'===Overall{"=" * 50}\n'
            f"Accuracy:[{all_acc_mean:.4f}±{all_acc_std:.4f}]\n"
            f"Micro-F1:[{all_microf_mean:.4f}±{all_microf_std:.4f}]\n"
            f"Macro-F1:[{all_macrof_mean:.4f}±{all_macrof_std:.4f}]\n"
        )

if __name__ == "__main__":
    from GraphCLIP.utils.args import Arguments
    from data_provider import *
    args = Arguments().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_exps = {
        "exp1": pretrain,
        "exp2": pretrain,
        "exp3": pretrain_exp3,
        "exp4": pretrain_exp4,
    }
    pretrain_dict = pretrain_exps[args.model_id]

    exps = {
        f"exp{i}": {
            "node": globals()[f"NC_exp{i}"],
            "edge": globals()[f"EC_exp{i}"],
            "graph": globals()[f"GC_exp{i}"],
        }
        for i in range(0, 5)
    }
    for name, dataset in exps[args.exp_id][args.task_name].items():
        main(args, pretrain_dict, name, dataset)