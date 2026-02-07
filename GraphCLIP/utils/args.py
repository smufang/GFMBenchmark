import argparse
from root import ROOT_DIR

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # Dataset
        # self.parser.add_argument('--dataset', type=str, help="dataset name", default='cora')
        # self.parser.add_argument('--source_data', type=str, help="dataset name", default='pubmed')
        # self.parser.add_argument('--target_data', type=str, help="dataset name", default='citeseer')
        
        # Model configuration
        self.parser.add_argument('--ckpt', type=str, help="the name of checkpoint", default='graphclip')
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.5)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu', 
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        # self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--last_activation', action='store_true', help="the last layer will use activation function or not")
        # self.parser.add_argument('--model', type=str, help="model name", default='GNN', 
        #                          choices=['GNN'])
        self.parser.add_argument('--norm', type=str, help="the type of normalization, id denotes Identity(w/o norm), bn is batchnorm, ln is layernorm", default='id', 
                                 choices=['id', 'bn', 'ln'])
        # self.parser.add_argument('--encoder', type=str, help="the type of encoder", default='GCN_Encoder', 
        #                          choices=['GCN_Encoder', 'GAT_Encoder', 'SAGE_Encoder', 'GIN_Encoder', 'MLP_Encoder', 'GCNII_Encoder'])
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam', 
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate for pretrain", default=1e-5)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=1e-5)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=30)
        self.parser.add_argument('--batch_size', type=int, help="the batch size", default=256)
        
        # Processing node attributes
        self.parser.add_argument('--llm', action='store_true', help="use the output of llm as node features")
        self.parser.add_argument('--peft', type=str, help="the type of peft", default='lora', 
                                 choices=['lora', 'prefix', 'prompt', 'adapter', 'ia3'])
        self.parser.add_argument('--lm_type', type=str, help="the type of lm", default='tiny', 
                                 choices=['tiny', 'sbert', 'deberta', 'bert', 'e5', 'llama2', 'llama3', 'llama2-14', 'qwen2', 'qwen2.5-0.5b', 'tiny', 'sbert2'])
        
        # used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=1)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling", default='rw', 
                                 choices=['rw', 'khop', 'shadow'])
    
        # prompt type
        self.parser.add_argument('--prompt', type=str, help="the type of prompt tuning", default='gppt', 
                                 choices=['gppt', 'graphprompt', 'prog', 'gpf'])

        # Exp
        self.parser.add_argument('--model', type=str, default='graphclip', help='model name')
        self.parser.add_argument('--model_id', type=str, default='exp1', help='pretrain model id', choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4'])
        self.parser.add_argument('--task_name', type=str, default='pretrain', help='task name: pretrain/node/edge/graph', choices=['pretrain', 'node', 'edge', 'graph'])
        self.parser.add_argument('--pattern', type=str, default='cross', help='pattern: cross-domain/single-domain/simple/no-pretrain', choices=['simple', 'cross', 'single','none'])
        self.parser.add_argument('--preprocess', type=str, default='basic', help='preprocessing method', choices=['basic', 'simple'])
        self.parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
        self.parser.add_argument('--seed', type=int, help="random seed", default=0)
        self.parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for data loading')
        self.parser.add_argument('--compress_function', type=str, default='none',help='dimension alignment method', choices=['pca', 'svd', 'svd_gcope', 'none'])
        self.parser.add_argument('--cache_compress', type=bool, default=False, help='whether to cache the compression results')
        self.parser.add_argument('--is_logging', type=bool, default=False, help='whether to log training progress')
        self.parser.add_argument('--continue_train', type=bool, default=False, help='whether to continue training from checkpoint')
        self.parser.add_argument('--checkpoints', type=str, default=str(ROOT_DIR / 'checkpoints'), help='checkpoint path')
        self.parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss criterion', choices=["cross_entropy", "mse", "nll"])

        self.parser.add_argument('--input_dim', type=int, help="the dimension of input features", default=384)
        self.parser.add_argument('--hidden_dim', type=int, help="the dimension of hidden features", default=1024)
        self.parser.add_argument('--output_dim', type=int, help="the dimension of output features", default=384)

        self.parser.add_argument('--exp_id', type=str, default='exp1', help='Exp id', choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4'])
        self.parser.add_argument('--num_shots', type=int, default=5, help='number of shots for few-shot learning')
        self.parser.add_argument('--num_tasks', type=int, default=50, help='number of tasks for few-shot learning')
        
    def parse_args(self):
        return self.parser.parse_args()