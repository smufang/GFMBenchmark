import torch
from torch_geometric.data import HeteroData, TemporalData, Data
from data_provider.data_generator import *

all={
    # Pretraining datasets
    'Cora': cora,
    'ogbn-arxiv': ogbn_arxiv,
    'ACM': acm,
    'DBLP': dblp,
    'Reddit': reddit,
    'Texas': texas,
    'Wisconsin': wisconsin,
    'Cornell': cornell,
    'IMDB': imdb,
    'Photo': photo,
    'Computers': computers,
    'Amazon': amazon,
    'Amazon-HeTGB': amazonh,
    'HIV': hiv,
    'COX2': cox2,
    'PROTEINS': proteins,
    'ENZYMES': enzymes,
    'FB15K-237': fb15k237,
    'NELL': nell,
    'Elliptic': elliptic,
    # Downstream datasets
    'Pubmed': pubmed,
    'ogbn-mag': ogbn_mag,
    'Wikipedia': wikipedia,
    'Actor': actor,
    'Chameleon': chameleon,
    'Products': products,
    'ogbn-proteins': ogbn_proteins,
    'T-Finance': tfinance,
    'DGraph': dgraph,
    'PCBA': pcba,
    'BZR': bzr,
    'WIKI': wiki,
    'WN18RR': wn18rr,
}

pretrain = {
    # 'Cora': cora,
    # 'ogbn-arxiv': ogbn_arxiv,
    # 'ACM': acm,
    # 'DBLP': dblp,
    # 'Reddit': reddit,
    # 'Texas': texas,
    'Wisconsin': wisconsin,
    # 'Cornell': cornell,
    # 'IMDB': imdb,
    # 'Photo': photo,
    # 'Computers': computers,
    # 'Amazon': amazon,
    # 'Amazon-HeTGB': amazonh,
    # 'HIV': hiv,
    # 'COX2': cox2,
    # 'PROTEINS': proteins,
    # 'ENZYMES': enzymes,
    # 'FB15K-237': fb15k237,
    # 'NELL': nell,
    # 'Elliptic': elliptic
}

pretrain_exp3 = {
    'Cora': cora,
    'ACM': acm,
    'DBLP': dblp,
}

pretrain_exp4 = {
    'Photo': photo,
    'Computers': computers,
    'COX2': cox2,
    'PROTEINS': proteins,
    'ENZYMES': enzymes,
    'Elliptic': elliptic
}

NC_exp1 = {
    'Pubmed': pubmed,
    'Wikipedia': wikipedia,
    'Actor': actor,
    'T-Finance': tfinance,
    'DGraph': dgraph,
    ## No label name ###
    'Chameleon': chameleon,
    'ogbn-proteins': ogbn_proteins,
    'ogbn-mag': ogbn_mag,
    'Products': products,
    }

EC_exp1 = {
    'WN18RR': wn18rr,
    ### No label name ###
    'DGraph': dgraph,
    'WIKI': wiki,
    }

GC_exp1 = {
    'BZR': bzr,
    ### No label name ###
    'PCBA': pcba,
    }

NC_exp2 = {
    # 'Cora': cora,
    # 'ACM': acm,
    # 'Reddit': reddit,
    'Wisconsin': wisconsin,
    # 'Elliptic': elliptic,
    # ## No label name ###
    # 'Photo': photo,
}

EC_exp2 = {
    'FB15K-237': fb15k237,
}

GC_exp2 = {
    'HIV': hiv,
    'COX2': cox2,
    'PROTEINS': proteins
}

NC_exp3 = {
    'Wikipedia': wikipedia,
    'Actor': actor,
    'T-Finance': tfinance,
    'DGraph': dgraph,
    ## No label name ###
    'Chameleon': chameleon,
    'ogbn-proteins': ogbn_proteins,
    'Products': products,
    ## cite
    'Cora': cora,
    'Pubmed': pubmed,
    'ACM': acm,
    'ogbn-mag': ogbn_mag,
}

EC_exp3 = {
    'WN18RR': wn18rr,
    ## No label name ###
    'DGraph': dgraph,
    'WIKI': wiki,
}

GC_exp3 = {
    'BZR': bzr,
    ### No label name ###
    'PCBA': pcba,
}

NC_exp4 = {
    'Pubmed': pubmed,
    'Wikipedia': wikipedia,
    'Actor': actor,
    # ### No label name ###
    'Chameleon': chameleon,
    'ogbn-mag': ogbn_mag,
    'Products': products,
}

EC_exp4 = {
    'WN18RR': wn18rr,
    ### No label name ###
    'WIKI': wiki,
}

GC_exp4 = {
    ### No label name ###
    'PCBA': pcba,
}

# exp0 is for testing
pretrain_exp0 = {
    'Cora': cora,
    'Citeseer': citeseer,
    #'Pubmed': pubmed,
    'Squirrel': squirrel,
    'Cornell': cornell,
    'Chameleon': chameleon
    }


NC_exp0 = {
    'Cora': cora,
    'CoraPYG':corapyg,
    'ACM': acm,
    'Photo': photo,
    }

EC_exp0 = {}
GC_exp0 = {}

TAGs = ['Cora','Pubmed', 'ogbn-arxiv', 
       'Actor', ' Texas', ' Wisconsin', ' Cornell', 
       'Amazon-HeTGB', 'Products', 
       'FB15K-237','WN18RR']


def dict2id(data_dict):
    return {name: idx for idx, name in enumerate(data_dict.keys())}



__all__ = ["torch", "HeteroData", "TemporalData", "Data", "dict2id", "TAGs",
           "all", "pretrain", "pretrain_exp3", "pretrain_exp4","pretrain_exp0",
           "NC_exp1", "EC_exp1", "GC_exp1",
           "NC_exp2", "EC_exp2", "GC_exp2",
           "NC_exp3", "EC_exp3", "GC_exp3",
           "NC_exp4", "EC_exp4", "GC_exp4",
           "NC_exp0", "EC_exp0", "GC_exp0"]
