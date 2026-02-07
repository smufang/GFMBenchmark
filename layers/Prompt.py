import torch
import torch.nn as nn
from typing import List

    
class TextPrompt(nn.Module):
    def __init__(self, hidden_dim, combinetype='mul'):
        super(TextPrompt, self).__init__()
        self.prompt= nn.Parameter(torch.FloatTensor(1,hidden_dim), requires_grad=True)
        self.combinetype = combinetype
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    def forward(self, graph_embedding):
        """
        graph_embedding: [num_nodes, hidden_dim]
        """
        # print(self.prompt)
        # assert not torch.isnan(self.prompt).any(), f"NaN in prompt"
        if self.combinetype == 'add':
            graph_embedding = graph_embedding + self.prompt
        elif self.combinetype == 'mul':
            graph_embedding= graph_embedding * self.prompt
        else:
            raise ValueError(f"Unknown combinetype: {self.combinetype}")
        return graph_embedding


class AlignPrompt(nn.Module):
    def __init__(self, hidden_dim, domain_id, combinetype='mul'):
        super(AlignPrompt, self).__init__()
        self.num_domains = len(domain_id)
        self.domain_id = domain_id
        self.prompt = nn.Parameter(torch.FloatTensor(self.num_domains, hidden_dim), requires_grad=True)
        self.combinetype = combinetype
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    def forward(self, graph_embedding, names: List[str]):
        """
        graph_embedding: [num_nodes, hidden_dim]
        names: [num_nodes]
        """
        # Check for unknown domain.
        unknown_names = set(name for name in names if name not in self.domain_id)
        if unknown_names:
            raise KeyError(f"Unknown domain name(s) in input: {unknown_names}")

        ids = [self.domain_id[name] for name in names]
        selected_prompt = self.prompt[ids]  # [num_nodes, hidden_dim]

        if self.combinetype == 'add':
            graph_embedding = graph_embedding + selected_prompt
        elif self.combinetype == 'mul':
            graph_embedding = graph_embedding * selected_prompt
        else:
            raise ValueError(f"Unknown combinetype: {self.combinetype}")

        return graph_embedding


class ComposedPrompt(nn.Module):
    def __init__(self, prompt_weights, combinetype='mul'):
        # prompt_weight: Prompt.weight.detach() [num_domains, hidden_dim]
        super(ComposedPrompt, self).__init__()
        num_domains = prompt_weights.size(0)
        self.compose_weight = nn.Parameter(torch.FloatTensor(1, num_domains), requires_grad=True)
        self.prompt_weights = prompt_weights
        self.combinetype = combinetype
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.compose_weight, a=0, b=1)

    def forward(self, graph_embedding, names: List[str]):
        """
        graph_embedding: [num_nodes, hidden_dim]
        names: [num_nodes] (not used here, but kept for interface consistency)
        """
        self.prompt_weights = self.prompt_weights.to(self.compose_weight.device)
        # [1, num_domains] @ [num_domains, hidden_dim] -> [1, hidden_dim]
        prompt = self.compose_weight @ self.prompt_weights
        if self.combinetype == 'add':
            graph_embedding = graph_embedding + prompt
        elif self.combinetype == 'mul':
            graph_embedding = graph_embedding * prompt
        else:
            raise ValueError(f"Unknown combinetype: {self.combinetype}")
        return graph_embedding



