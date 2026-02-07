import argparse
import torch
import random
import time
from tqdm import tqdm
from G2P2.model import CLIP, tokenize
from exp.exp_pretrain_tag import ExpPretrain
from data_provider import *
from utils.tools import EarlyStopping
import os
import time


def build_neigh_ids(num_nodes, edge_index, num_neighs=3) -> dict:
    neighs = {}
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        # avoid self-loop
        if u != v:
            neighs.setdefault(u, set()).add(v)

    neigh_ids = torch.zeros((num_nodes, num_neighs), dtype=torch.long)
    for node, ts in neighs.items():
        ts = random.choices(list(ts), k=num_neighs)
        neigh_ids[node] = torch.tensor(ts, dtype=torch.long)
    return neigh_ids


def main(args):
    args.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    pretrain_exps = {'exp1': pretrain, 'exp2': pretrain, 'exp3': pretrain_exp3, 'exp4': pretrain_exp4}
    pretrain_dict = pretrain_exps[args.model_id]
    exp = ExpPretrain(args, pretrain_dict=pretrain_dict)
    path = exp._create_save_path()

    model = CLIP(args).to(exp.device)

    os.makedirs(path, exist_ok=True)
    if args.continue_train is True:
        model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(model_path):
            exp._load_checkpoint(model_path, model)
            exp._print_main(f'Loaded checkpoint from {model_path}')
        else:
            exp._print_main(f"Checkpoint path {model_path} does not exist.")

    exp._print_main(f'Checkpoints path: {path}')

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    time_start = time.time()
    model.train()

    for epoch in range(args.epochs):
        avg_loss = float("inf")
        train_loss = 0.0
        tot_num_task_nodes = 0
        epoch_time = time.time()
        train_loader = exp._get_loader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch_data in enumerate(pbar, start=1):
            batch_data = batch_data.to(exp.device)
            neigh_ids = build_neigh_ids(
                batch_data.num_nodes, batch_data.edge_index, num_neighs=args.num_neighs
            ).to(exp.device)
            if hasattr(batch_data, 'token_cache') and batch_data.token_cache is not None:
                # save time for tokenization by using cached tokens
                text_tokens = batch_data.token_cache.to(exp.device)
            else:
                text_tokens = tokenize(batch_data.raw_texts, context_length=args.context_length).to(exp.device)

            loss = model.forward(
                batch_data, text_tokens, neigh_ids, device=exp.device, training=True
            )

            num_task_nodes = len(batch_data.input_nodes)
            tot_num_task_nodes += num_task_nodes
            train_loss += loss * num_task_nodes
            avg_loss = train_loss / tot_num_task_nodes
            torch.cuda.empty_cache()
            # Update progress bar and log (only in main process)
            if exp._is_main_process() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    {"loss": f"{loss:.6f}", "avg_loss": f"{avg_loss:.6f}"}
                )

        exp._write_log(
            f"Epoch {epoch + 1} finished. "
            f"Average Loss: {avg_loss:.6f}, "
            f"Time: {time.time() - epoch_time:.2f}s"
        )

        early_stopping(avg_loss, model, path)
        if early_stopping.early_stop:
            exp._print_main(f"Early stopping at epoch: {epoch + 1}")
            exp._write_log(f"Early stopping at epoch: {epoch + 1}")
            break

    best_model_path = os.path.join(path, "checkpoint.pth")
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


if __name__ == "__main__":
    from root import ROOT_DIR
    parser = argparse.ArgumentParser(description='Graph Foundation Model')
    # G2P2 parameters
    #parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epochs", type=int, default=3, help="epoch number")
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    #parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--num_neighs", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    #parser.add_argument("--gpu", type=int, default=0)
    #parser.add_argument("--data_name", type=str, default="cora")

    # exp parameters
    parser.add_argument('--model', type=str, default='g2p2', help='model name')
    parser.add_argument('--model_id', type=str, default='exp1', help='pretrain model id')
    parser.add_argument('--task_name', type=str, default='pretrain',
                        help='task name: pretrain/node/edge/graph')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Device parameters
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Training parameters
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10, 10, 10], help='number of neighbors for sampling')
    parser.add_argument('--max_nodes', type=int, default=60000, help='maximum number of nodes per batch')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for sampling')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--compress_function', type=str, default='pca',
                        help='dimension alignment method: pca/svd/svd_gcope/none')
    parser.add_argument('--cache_compress', type=bool, default=False, help='whether to cache the compression results')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=128, help='same with gnn_input')

    parser.add_argument('--checkpoints', type=str, default=str(ROOT_DIR / 'checkpoints'), help='checkpoint path')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training from last checkpoint')
    parser.add_argument('--is_logging', type=bool, default=False, help='whether to log training progress')

    parser.add_argument('--pattern', type=str, default='cross', help='pattern: cross-domain/single-domain/simple/no-pretrain', choices=['simple', 'cross', 'single','none'])
    parser.add_argument('--preprocess', type=str, default='basic', help='preprocessing method', choices=['basic', 'simple'])

    args = parser.parse_args()
    
    main(args)