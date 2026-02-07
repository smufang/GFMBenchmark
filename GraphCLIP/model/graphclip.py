from transformers import AutoModel
import numpy as np
import torch
from .gt import GPS



#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



text_ids = {
    'tiny': 'sentence-transformers/all-MiniLM-L6-v2',
    'sbert':  'sentence-transformers/multi-qa-distilbert-cos-v1', #'sentence-transformers/all-MiniLM-L6-v2', #'sentence-transformers/multi-qa-distilbert-cos-v1',
    'e5': 'intfloat/e5-base-v2',
    'deberta': 'microsoft/deberta-v3-base',
}



class Config:
    def __init__(self):
        self.prefix_projection=True
        self.pre_seq_len=10
        self.num_hidden_layers=6
        self.prefix_hidden_size=384
        self.hidden_size=384

config = Config()

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class GraphCLIP(torch.nn.Module):
    def __init__(self, graph_input_dim, graph_hid_dim, graph_num_layer, attn_kwargs, text_model='tiny'):
        super().__init__()
        self.graph_model = GPS(in_dim=graph_input_dim, channels=graph_hid_dim, out_dim=graph_hid_dim, 
                               pe_dim=8, num_layers=graph_num_layer, attn_type='multihead', attn_kwargs=attn_kwargs)
        self.text_model_type = text_model
        text_id = text_ids[text_model]
        text_model = AutoModel.from_pretrained(text_id)
        self.text_model = text_model
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # for prompt tuning
        self.contrast_mode = 'all'
        self.temperature = 0.2
        self.base_temperature = 0.2
        self.pre_seq_len=config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

    # use this instead of encode_text for prompt tuning
    # def prompt_text(self, input_ids, token_type_ids, attention_mask):
    #     batch_size = input_ids.shape[0]

    #     # 1) 生成前缀嵌入（隐藏维度）
    #     prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.text_model.device)
    #     prefix_embs = self.prefix_encoder.embedding(prefix_tokens)  # [B, pre_len, hidden]

    #     # 2) 注意力掩码
    #     prefix_mask = torch.ones(batch_size, self.pre_seq_len, dtype=attention_mask.dtype, device=self.text_model.device)
    #     attention_mask_prefix = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, pre_len+L]

    #     # 3) 词嵌入拼接
    #     input_embs = self.text_model.embeddings.word_embeddings(input_ids)  # [B, L, hidden]
    #     combined_embeds = torch.cat([prefix_embs, input_embs], dim=1)      # [B, pre_len+L, hidden]

    #     # 4) token_type_ids 同步扩展（如模型支持且非 None）
    #     if token_type_ids is not None:
    #         prefix_types = torch.zeros(batch_size, self.pre_seq_len, dtype=token_type_ids.dtype, device=self.text_model.device)
    #         token_type_ids = torch.cat([prefix_types, token_type_ids], dim=1)  # [B, pre_len+L]

    #     # 5) 前向
    #     text_output = self.text_model(
    #         inputs_embeds=combined_embeds,
    #         attention_mask=attention_mask_prefix,
    #         token_type_ids=token_type_ids
    #     )

    #     text_embs = mean_pooling(text_output.last_hidden_state, attention_mask_prefix)
    #     return text_embs
    
    # def get_prompt_embeddings(self, batch_size):
    #     prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.text_model.device)
    #     past_key_values = self.prefix_encoder(prefix_tokens) # [batch, prefix_len, hidden]
    #     return past_key_values

    def prompt_text(self, input_ids, token_type_ids, attention_mask):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.text_model.device)
        attention_mask_prefix = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if self.text_model_type in ['xxx']:
            text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask_prefix, past_key_values=past_key_values)
            text_embs = text_output[0][:,0,:].squeeze(1)
        else:
            text_output = self.text_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask_prefix, past_key_values=past_key_values)
            text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.text_model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            6 * 2, 
            12, # head
            384//12
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def encode_graph(self, batch):
        graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs, center_embs

    def encode_text(self, input_ids, token_type_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def forward(self, batch_g, batch_t):
        graph_features, c_features = self.encode_graph(batch_g)
        text_features = self.encode_text(**batch_t)

        # normalized features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_graph, logits_per_text
    
    def freeze_text(self):
        for k, v in self.text_model.named_parameters():
            v.requires_grad = False

    def freeze_graph(self):
        for k, v in self.graph_model.named_parameters():
            v.requires_grad = False
    
    # used for prompt tuning
    def sup_loss(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    