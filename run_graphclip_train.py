import torch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from GraphCLIP.model import GraphCLIP
from GraphCLIP.utils.augmentation import adversarial_aug_train, graph_aug
from GraphCLIP.utils.args import Arguments
from GraphCLIP.utils.process import parse_source_data
from GraphCLIP.model.dp import TextCLIP, GCLIP, calculate_loss, create_logits
from GraphCLIP.generate_feature import *
from utils.tools import EarlyStopping


def train(data_loader, epoch):
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
    for i, batch in enumerate(pbar, start=1):
        optimizer.zero_grad()
        model.train()

        model.graph_model.redraw_projection.redraw_projections()
        summaries = [g.summary for g in batch]
        batch_t = tokenizer(summaries, truncation=True, padding=True, return_tensors="pt", max_length=512)
        # DP codes
        batch = [graph_aug(g, 0.3, 0.2) for g in batch]

        def node_attack(perturbs):
            for b_id, g in enumerate(batch):
                g.x = g.x + perturbs[b_id]
            graph_embs, _ = model_graph(batch)
            text_embs = model_text(input_ids=batch_t['input_ids'], token_type_ids=None,
                                   attention_mask=batch_t['attention_mask'])
            logit_scale = model.logit_scale.exp()
            logits_per_graph, logits_per_text = create_logits(graph_embs, text_embs, logit_scale)
            loss = calculate_loss(logits_per_graph, logits_per_text, criterion)

            return loss

        perturb_shapes = [g.x.shape for g in batch]
        loss = adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, 1e-2, 3)
        loss.backward()
        total_loss += loss.item() * len(batch)
        optimizer.step()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    return total_loss / len(data_loader.dataset)


if __name__ == "__main__":
    args = Arguments().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_exps = {
        "exp1": pretrain,
        "exp2": pretrain,
        "exp3": pretrain_exp3,
        "exp4": pretrain_exp4,
    }
    pretrain_dict = pretrain_exps[args.model_id]
    exp = ExpPretrainGraphCLIP(args, pretrain_dict)
    os.makedirs(exp.save_path, exist_ok=True)

    attn_kwargs = {'dropout': 0.0}
    text_model = args.lm_type
    model = GraphCLIP(args.input_dim, args.hidden_dim, 12, attn_kwargs, text_model=text_model)

    # freeze text model
    model.freeze_text()

    # DP codes
    model_text = TextCLIP(model)
    model_graph = GCLIP(model)
    model_text = torch.nn.DataParallel(model_text)  # use torch DP
    model_graph = DataParallel(model_graph)  # use pyg DP
    model.to(args.device)

    text_ids = {
        'tiny': 'sentence-transformers/all-MiniLM-L6-v2',
        'sbert': 'sentence-transformers/multi-qa-distilbert-cos-v1',
        'e5': 'intfloat/e5-base-v2'
    }

    tokenizer = AutoTokenizer.from_pretrained(text_ids[text_model])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # collect source data
    all_source_graph = []
    for name, data in exp.pretrain_dict.items():
        is_tag = True
        # This is an analysis to the necessity of text-free graphs' LLM ouput as summary
        # if name not in TAGs:
        #     is_tag = False
        source_graph = parse_source_data(name, data, is_tag=is_tag)
        all_source_graph.extend(source_graph)

    exp._print_main(f"We have {len(all_source_graph)} pretraining graphs")
    exp._write_log(f"We have {len(all_source_graph)} pretraining graphs")

    train_loader = DataListLoader(all_source_graph, batch_size=args.batch_size,
                                  shuffle=True)  # use DataListLoader for DP rather than DataLoader
    
    exp._print_main(f"Let's use {torch.cuda.device_count()} GPUs!")
    exp._write_log(f"Let's use {torch.cuda.device_count()} GPUs!")

    if args.continue_train:
        checkpoint_path = os.path.join(exp.save_path, "checkpoint.pth")
        exp._load_checkpoint(checkpoint_path, model)
        exp._print_main(f"Continue training from {checkpoint_path}")
        exp._write_log(f"Continue training from {checkpoint_path}")

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    time_start = time.time()
    for epoch in range(args.epochs):
        train_loss = 0.0
        tot_num_task_nodes = 0
        epoch_time = time.time()
        avg_loss = train(train_loader, epoch=epoch)
        torch.cuda.empty_cache()

        exp._write_log(
            f"Epoch {epoch + 1} finished. "
            f"Average Loss: {avg_loss:.6f}, "
            f"Time: {time.time() - epoch_time:.2f}s"
        )

        early_stopping(avg_loss, model, exp.save_path)
        if early_stopping.early_stop:
            exp._print_main(f"Early stopping at epoch: {epoch + 1}")
            exp._write_log(f"Early stopping at epoch: {epoch + 1}")
            break

    best_model_path = os.path.join(exp.save_path, "checkpoint.pth")
    exp._load_checkpoint(best_model_path, model)

    training_info = {
        "best_loss": early_stopping.val_loss_min,
        "training_time": time.time() - time_start,
        "model_path": best_model_path,
    }

    exp._print_main(
        f"\nTraining finished:",
        f'Best loss: {training_info["best_loss"]:.6f} ',
        f'Training time: {training_info["training_time"]:.2f}s',
        f'\nSave path: {training_info["model_path"]}',
    )
    exp._write_log(
        f"Training finished:"
        f'Best loss: {training_info["best_loss"]:.6f} '
        f'Training time: {training_info["training_time"]:.2f}s\n'
        f'Save path: {training_info["model_path"]}'
    )
