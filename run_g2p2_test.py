import argparse
import torch
from G2P2.model import CLIP
from G2P2.model_g_coop import CoOp
from exp.exp_downstream_tag import ExpDownstreamBatch
from data_provider import *


def main(args, pretrain_dict, name, dataset):
    args.device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    exp = ExpDownstreamBatch(
        args, pretrain_dict=pretrain_dict, name=name, dataset=dataset
    )
    pretrain_model = exp._get_pretrain_model(CLIP(args))
    if args.task_name == "edge":
        classnames = exp.data.relation_texts
    else:
        classnames = [f"{name}{desc}" for name, desc in zip(exp.data.label_names, exp.data.label_descs)]
    tuning_model = CoOp(
        args,
        classnames=classnames,
        clip_model=pretrain_model,
        g_texts=None,
        device=exp.args.device,
    )
    learnable_params = list(tuning_model.model.prompt_learner.parameters()) + list(tuning_model.model.edge_linear.parameters())
    exp.run_tasks(tuning_model, learnable_params)


if __name__ == "__main__":
    from root import ROOT_DIR

    parser = argparse.ArgumentParser(description="Graph Foundation Model")

    # G2P2 parameters (downstream)
    parser.add_argument("--coop_n_ctx", type=int, default=4, help="number of prompt tokens")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for prompt tuning")

    parser.add_argument("--position", type=str, default="end", help="position of the prompt tokens")
    parser.add_argument("--class_specific", type=bool, default=False, help="if False, use same prompt for all classes")
    parser.add_argument("--ctx_init", type=bool, default=False, help="whether to initialize the prompt vectors")

    # G2P2 parameters (pretrain)
    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epochs", type=int, default=3, help="epoch number")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=2e-5, help="pretrain learning rate")
    parser.add_argument("--edge_coef", type=float, default=0.1)
    parser.add_argument("--num_neighs", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)

    # exp parameters
    parser.add_argument("--model", type=str, default="g2p2", help="model name")
    parser.add_argument("--model_id", type=str, default="exp1", help="pretrain model id")
    parser.add_argument('--exp_id', type=str, default='exp1', help='Exp id')
    parser.add_argument("--task_name", type=str, default="other-simple", help="task name: pretrain/node/edge/graph")
    parser.add_argument('--pattern', type=str, default='cross', help='pattern: cross-domain/single-domain/simple/no-pretrain', choices=['simple', 'cross', 'single','none'])
    parser.add_argument('--preprocess', type=str, default='basic', help='preprocessing method', choices=['basic', 'simple'])
    
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # Device parameters
    parser.add_argument("--use_gpu", type=bool, default=False, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision')

    # Training parameters
    parser.add_argument("--max_nodes", type=int, default=60000, help="maximum number of nodes per batch")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for sampling")
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--compress_function",type=str,default="pca",help="dimension alignment method: pca/svd/svd_gcope/none")
    parser.add_argument("--cache_compress",type=bool,default=False,help="whether to cache the compression results")
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss criterion', choices=["cross_entropy", "mse", "nll"])
    # Model parameters
    parser.add_argument("--input_dim", type=int, default=128, help="same with gnn_input")

    # Few-shot learning parameters
    parser.add_argument('--num_shots', type=int, default=5, help='number of shots for few-shot learning')
    parser.add_argument('--num_tasks', type=int, default=50, help='number of tasks for few-shot learning')

    parser.add_argument("--checkpoints",type=str,default=str(ROOT_DIR / "checkpoints"),help="checkpoint path")
    parser.add_argument("--continue_train",type=bool,default=False,help="continue training from last checkpoint")
    parser.add_argument("--is_logging",type=bool,default=False,help="whether to log training progress")

    args = parser.parse_args()

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
