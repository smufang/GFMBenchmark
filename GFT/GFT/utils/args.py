import argparse


def get_args_pretrain():
    parser = argparse.ArgumentParser('Pretrain')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=int, default=3)

    # Encoder Parameters
    parser.add_argument('--input_dim', type=int, default=50) #exp1:64
    parser.add_argument('--hidden_dim', type=int, default=50) #exp1:64
    parser.add_argument('--num_layers', '--layers', type=int, default=2)
    parser.add_argument('--activation', '--act', type=str, default="relu")
    parser.add_argument('--backbone', type=str, default="sage")
    parser.add_argument('--normalize', type=str, default="batch", choices=['none', 'batch', 'layer'])
    parser.add_argument('--dropout', type=float, default=0.15)

    # VQ Parameters
    parser.add_argument('--code_dim', type=int, default=50) #exp1:64
    parser.add_argument('--codebook_size', type=int, default=64) #exp1:128
    parser.add_argument('--codebook_head', type=int, default=4)
    parser.add_argument('--codebook_decay', type=float, default=0.8)
    parser.add_argument('--commit_weight', type=float, default=10)
    parser.add_argument('--ortho_reg_weight', type=float, default=1)
    parser.add_argument('--ortho_reg_max_codes', type=int, default=32)

    # Pretrain Dataset
    parser.add_argument('--compress_function', type=str, default='pca',
                        help='dimension alignment method: pca/svd')
    parser.add_argument('--pretrain_epochs', '--pt_epochs', '--epochs', type=int, default=50)
    parser.add_argument('--pretrain_lr', '--pt_lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_weight_decay', '--pt_decay', '--decay', type=float, default=1e-5)
    parser.add_argument('--pretrain_batch_size', '--pt_batch', type=int, default=1024)
    parser.add_argument('--feat_p', type=float, default=0.2)
    parser.add_argument('--edge_p', type=float, default=0.2)
    parser.add_argument('--topo_recon_ratio', type=float, default=0.1)
    parser.add_argument('--feat_lambda', type=float, default=100)
    parser.add_argument('--topo_lambda', type=float, default=0.01)
    parser.add_argument('--topo_sem_lambda', type=float, default=100)
    parser.add_argument('--sem_lambda', type=float, default=1)
    parser.add_argument('--sem_encoder_decay', type=float, default=0.99)
    parser.add_argument('--use_schedular', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    args = parser.parse_args()
    return vars(args)


def get_args_finetune():
    parser = argparse.ArgumentParser('Finetune')

    # General Parameters
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument("--setting", type=str, default="few_shot") #standard/few_shot
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=int, default=3)

    # Few-shot Parameters
    parser.add_argument('--compress_function', type=str, default='pca',
                        help='dimension alignment method: pca/svd')
    parser.add_argument("--task_name", type=str, default="node") #edge, graph
    parser.add_argument("--n_task", type=int, default=50)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--n_query", type=int, default=3)

    # Pre-train Parameters
    parser.add_argument("--pretrain_model_epoch", '--pt_epochs', type=int, default=49)
    parser.add_argument('--pretrain_seed', '--pt_seed', type=int, default=42)

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", '--act', type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # VQ Parameters
    parser.add_argument("--code_dim", type=int, default=50)
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--codebook_head", type=int, default=4)
    parser.add_argument("--codebook_decay", type=float, default=0.8)
    parser.add_argument("--commit_weight", type=float, default=0.25)
    parser.add_argument("--ortho_reg_weight", type=float, default=1)
    parser.add_argument("--ortho_reg_max_codes", type=int, default=32)

    # Fine-Tune Parameters
    parser.add_argument("--finetune_dataset", "--dataset", "--data", type=str, default="Pubmed")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--finetune_epochs", "--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--finetune_lr", "--lr", type=float, default=1e-3)
    parser.add_argument("--query_node_code_first", action="store_true", help="Use node code to form edge/graph code")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision')

    # Model Parameters
    parser.add_argument("--separate_decoder_for_each_head", type=bool, default=True)
    parser.add_argument("--use_z_in_predict", type=bool, default=True)
    parser.add_argument("--use_cosine_sim", type=bool, default=True)
    parser.add_argument("--lambda_proto", type=float, default=1)
    parser.add_argument("--lambda_act", type=float, default=1)
    parser.add_argument("--trade_off", type=float, default=0.5)
    parser.add_argument("--num_instances_per_class", type=int, default=20)
    parser.add_argument('--no_lin_clf', action='store_true')
    parser.add_argument('--no_proto_clf', action='store_true')

    args = parser.parse_args()
    return vars(args)
