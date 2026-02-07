import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from layers import *
from utils.similarity import cosine_similarity


class Pretrain(nn.Module):
    def __init__(self, configs):
        super(Pretrain, self).__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim
        num_layers = configs.num_layers
        num_heads = configs.num_heads
        backbone = configs.backbone
        combinetype = configs.combinetype
        activation = configs.activation
        dropout = configs.dropout
        self.domain_id = configs.pretrain_domain_id
        self.mode = configs.mode
        self.temperature = configs.temperature
        self.drop_percent = configs.drop_percent
        self.aug_type = configs.aug_type
        self.num_neg_samples = configs.num_neg_samples

        if backbone == 'gcn':
            self.gnn_enc = GCNLayers(input_dim, hidden_dim, num_layers, dropout=dropout, activation=activation)
        elif backbone == 'gat':
            self.gnn_enc = GATLayers(input_dim, hidden_dim, num_layers, num_heads=num_heads, concat=True, bias=True,
                                     dropout=dropout, activation=activation)

        self.lp = LpGPT()
        self.graphcl = GraphCLGPT(hidden_dim, sim='bil')

        self.feature_prompt_layer = AlignPrompt(input_dim, domain_id=self.domain_id, combinetype=combinetype)
        self.loss = nn.BCEWithLogitsLoss()

    def get_domain_ids(self):
        return self.domain_id

    def compute_linkpred_prelogits(self, data):
        data_fp = data.clone()
        data_fp.x = self.feature_prompt_layer(data_fp.x, data_fp.name)
        fea_prelogits = self.lp(self.gnn_enc, data_fp, prompt_layers=None)
        return fea_prelogits

    def compute_graphcl_prelogits(self, data, aug_type='edge',
                                  samp_bias1=None, samp_bias2=None):
        data_fp = data.clone()
        data_fp.x = self.feature_prompt_layer(data_fp.x, data_fp.name)
        fea_prelogits, fea_lbl = self.graphcl(self.gnn_enc, data_fp, samp_bias1, samp_bias2, aug_type=aug_type,
                                              drop_percent=self.drop_percent, prompt_layers=None)

        return fea_prelogits, fea_lbl

    def get_embeddings(self, data):
        if self.mode == 'lp':
            h = self.gnn_enc(data.x, data.edge_index, LP=True)
        else:
            h = self.gnn_enc(data.x, data.edge_index, LP=False)
        c = torch.mean(h, 0)
        return h.detach(), c.detach()

    def get_prompt_weights(self):
        return {
            "feature_prompt": self.feature_prompt_layer.prompt.detach()
        }

    def forward(self, data, samp_bias1=None, samp_bias2=None):
        if self.mode == 'gcl':
            logits, lbl = self.compute_graphcl_prelogits(data, aug_type=self.aug_type,
                                                         samp_bias1=samp_bias1, samp_bias2=samp_bias2
                                                         )
            total_loss = self.loss(logits / self.temperature, lbl)
        elif self.mode == 'lp':
            logits = self.compute_linkpred_prelogits(data)
            sample_edge_index = LpGPT.prompt_pretrain_sample(data.edge_index, data.name, data.num_nodes,
                                                             n=self.num_neg_samples)
            total_loss = LpGPT.compare_loss(logits, sample_edge_index, temperature=self.temperature)
        else:
            raise ValueError("Invalid mode. Use 'lp' for link prediction or 'gcl'")

        return total_loss


class Downstream(nn.Module):
    def __init__(self, configs, pretrain_model, dataset):
        super(Downstream, self).__init__()
        combinetype = configs.combinetype
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim
        downstream_domain_id = {configs.target_data: 0}

        self.task_name = configs.task_name
        self.num_classes = configs.num_classes
        self.temperature = configs.temperature
        
        self.combine_weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.gnn_enc = pretrain_model.gnn_enc
        for param in self.gnn_enc.parameters():
            param.requires_grad = False

        prompt_weights = pretrain_model.get_prompt_weights()
        assert 'feature_prompt' in prompt_weights, "feature_prompt weights must be provided for MDGPT downstream tasks."
        feature_prompt_weights = prompt_weights['feature_prompt']
        
        self.fea_holistic_prompts = AlignPrompt(input_dim, domain_id=downstream_domain_id, combinetype=combinetype)
        self.fea_specific_prompts = ComposedPrompt(feature_prompt_weights, combinetype=combinetype)
        class_dim = hidden_dim*2 if self.task_name == 'edge' else hidden_dim
        self.class_prototypes = torch.FloatTensor(self.num_classes, class_dim)
        self.init_weights()
        # self.f = nn.Linear(hidden_dim, self.num_classes)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.combine_weight)

    def embedding_forward(self, data):
        fea_hol_embed = self.fea_holistic_prompts(data.x, data.name)
        fea_spe_embed = self.fea_specific_prompts(data.x, data.name)
        fea_al_embed = self.act(
            self.combine_weight[0][0] * fea_hol_embed
            + self.combine_weight[0][1] * fea_spe_embed
        )
        fea_al_embed = self.gnn_enc(fea_al_embed, data.edge_index, prompt_layers=None, name=data.name)

        return fea_al_embed

    def compute_predictions(self, embed, labels=None, is_train=True):
        """
        embed: [num_items, hidden_dim]
        labels: [num_items] only used if is_train=True
        """
        if is_train and labels is not None:
            self.class_prototypes = torch_scatter.scatter(src=embed, index=labels, dim=0,
                        reduce='mean', dim_size=self.num_classes)  # [num_classes, hidden_dim]

        similarity = cosine_similarity(embed, self.class_prototypes, temperature=self.temperature, eps=1e-10)  # [num_items, num_classes]
        if is_train:
            return similarity
        else:
            return F.softmax(similarity, dim=1)

    def node_classification(self, data, labels, node_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        select_node_embed = node_embed[node_idx]  # [num_nodes, hidden_dim] -> selected
        return self.compute_predictions(select_node_embed, labels, is_train)

    def edge_classification(self, data, edge_type, edge_idx, is_train=True):
        # edge_idx: [2, num_task_edges]
        node_embed = self.embedding_forward(data)
        select_src_idx, select_dst_idx = edge_idx
        select_edge_embed = torch.cat([node_embed[select_src_idx], node_embed[select_dst_idx]], dim=-1)
        return self.compute_predictions(select_edge_embed, edge_type, is_train)

    def graph_classification(self, data, graph_labels, graph_idx, is_train=True):
        node_embed = self.embedding_forward(data)
        graph_embed = torch_scatter.scatter(node_embed, data.batch, dim=0, reduce='mean')
        selected_graph_embed = graph_embed[graph_idx]
        return self.compute_predictions(selected_graph_embed, graph_labels, is_train)

    def forward(self, data, labels, idx, is_train=True):
        if self.task_name == 'node':
            return self.node_classification(data, labels, idx, is_train)
        elif self.task_name == 'edge':
            return self.edge_classification(data, labels, idx, is_train)
        elif self.task_name == 'graph':
            return self.graph_classification(data, labels, idx, is_train)
        else:
            raise ValueError("Invalid task name. Use 'node', 'edge', or 'graph'.")