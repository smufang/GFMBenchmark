import argparse
import torch.distributed as dist
from exp.exp_pretrain import ExpPretrain
from exp.exp_pretrain_single import ExpPretrainSingle
from exp.exp_downstream_batch import ExpDownstreamBatch
from root import ROOT_DIR
from data_provider import *
import gc


def main():
    parser = argparse.ArgumentParser(description='Graph Foundation Model')

    # Basic parameters
    parser.add_argument('--model', type=str, default='samgpt', help='model name')
    parser.add_argument('--model_id', type=str, default='exp1', help='pretrain model id', choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4', "none"])
    parser.add_argument('--task_name', type=str, default='pretrain', help='task name: pretrain/node/edge/graph', choices=['pretrain', 'node', 'edge', 'graph'])
    parser.add_argument('--exp_id', type=str, default='exp1', help='Exp id', choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4'])
    parser.add_argument('--pattern', type=str, default='cross', help='pattern: cross-domain/single-domain/simple/no-pretrain', choices=['simple', 'cross', 'single','none'])
    parser.add_argument('--preprocess', type=str, default='basic', help='preprocessing method', choices=['basic', 'simple'])
    parser.add_argument('--mode', type=str, default='lp', help='pretrain task: lp/gcl')
    parser.add_argument('--backbone', type=str, default='gat', help='graph encoder: gcn/gat')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Device parameters
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='cuda/mps')
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Training parameters
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10, 10, 10], help='number of neighbors for sampling')
    parser.add_argument('--max_nodes', type=int, default=60000, help='maximum number of nodes per batch')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for sampling')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--activation', type=str, default='prelu', help='activation function: relu/prelu/gelu')
    parser.add_argument('--compress_function', type=str, default='pca',help='dimension alignment method', choices=['pca', 'svd', 'svd_gcope', 'none'])
    parser.add_argument('--cache_compress', type=bool, default=False, help='whether to cache the compression results')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss criterion', choices=["cross_entropy", "mse", "nll"])

    # Model parameters
    parser.add_argument('--combinetype', type=str, default='none', help='textprompt in MDGPT/SAMGPT/MDGFM', choices=['add', 'mul', 'none'])
    parser.add_argument('--input_dim', type=int, default=50, help='input dimension after dimension alignment')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--edge_dim', type=int, default=0, help='edge embedding dimension for Hetero Methods')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter')
    parser.add_argument('--num_heads', type=int, default=0, help='number of attention heads:gat only')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for contrastive loss')

    # Graph Contrastive Learning
    parser.add_argument('--aug_type', type=str, default='edge', help='augmentation type for lp', choices=['edge', 'mask', 'node', 'subgraph'])
    parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percentage for node/edge features')

    # Link Prediction
    parser.add_argument('--num_neg_samples', type=int, default=50,
                        help='number of negative samples for link prediction')

    parser.add_argument('--checkpoints', type=str, default=str(ROOT_DIR / 'checkpoints'), help='checkpoint path')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training from last checkpoint')
    parser.add_argument('--is_logging', type=bool, default=False, help='whether to log training progress')

    # Few-shot learning parameters
    parser.add_argument('--num_shots', type=int, default=5, help='number of shots for few-shot learning')
    parser.add_argument('--num_tasks', type=int, default=50, help='number of tasks for few-shot learning')

    # MDGFM
    parser.add_argument('--k', type=int, default=30, help='number of neighbors for k-NN graph construction')

    # GAT GCN
    parser.add_argument('--using_projection', type=bool, default=False, help='whether to use projection head for downstream tasks')

    args = parser.parse_args()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if dist.is_initialized():
        dist.destroy_process_group()

    pretrain_exps = {'exp1': pretrain, 
                     'exp2': pretrain, 
                     'exp3': pretrain_exp3, 
                     'exp4': pretrain_exp4, 
                     'exp0': pretrain_exp0,
                     'none': None}
    pretrain_dict = pretrain_exps[args.model_id]

    if args.task_name == 'pretrain':
        if args.pattern == 'cross':
            # cross-domain pretraining
            exp = ExpPretrain(args, pretrain_dict=pretrain_dict)
        elif args.pattern == 'single':
            # single-domain pretraining
            exp = ExpPretrainSingle(args, pretrain_dict=pretrain_dict)
        else:
            raise ValueError("Only 'cross' and 'single' patterns are supported for pretraining.")
        # Train model
        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print('Start training...')
        exp.train()

    if args.task_name == 'view':
        exp = ExpPretrain(args, pretrain_dict=pretrain_dict)
        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print('View pretraining...')
        exp.vali()
        

    if args.task_name in ['node', 'edge', 'graph']:
        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print(f'Starting {args.task_name.capitalize()} Classification Task...')

        exps = {
            f"exp{i}": {
                "node": globals()[f"NC_exp{i}"],
                "edge": globals()[f"EC_exp{i}"],
                "graph": globals()[f"GC_exp{i}"],
            }
            for i in range(0, 5)
        }

        for name, dataset in exps[args.exp_id][args.task_name].items():
            exp = ExpDownstreamBatch(args, pretrain_dict=pretrain_dict, name=name, dataset=dataset)
            exp.run_tasks()
            
            del exp
            torch.cuda.empty_cache()
            gc.collect()

        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print(f'All {args.task_name.capitalize()} Classification Tasks Completed.')

    if args.use_multi_gpu and dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


if __name__ == '__main__':
    main()
