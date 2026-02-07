import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import GCNConv


class SimpleHGNConv(MessagePassing):
    """
    PyG implementation of Simple-HGN Conv, aligned with DGL's myGATConv.
    Supports edge type embeddings and residual attention.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 num_etypes,
                 edge_dim,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 bias=False,
                 alpha=0.0):
        super().__init__(aggr='add', node_dim=0)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.alpha = alpha
        self.activation = activation

        # Node projection
        if isinstance(in_channels, tuple):
            self.fc_src = nn.Linear(in_channels[0], out_channels * num_heads, bias=False)
            self.fc_dst = nn.Linear(in_channels[1], out_channels * num_heads, bias=False)
        else:
            self.fc = nn.Linear(in_channels, out_channels * num_heads, bias=False)

        # Edge type embedding and projection
        self.edge_emb = nn.Embedding(num_etypes, edge_dim)
        self.fc_e = nn.Linear(edge_dim, edge_dim * num_heads, bias=False)

        # Attention parameters: [1, H, F] or [1, H, D]
        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.attn_e = nn.Parameter(torch.Tensor(1, num_heads, edge_dim))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Residual connection logic
        if residual:
            in_dst = in_channels[1] if isinstance(in_channels, tuple) else in_channels
            if in_dst != out_channels * num_heads:
                self.res_fc = nn.Linear(in_dst, num_heads * out_channels, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        # Bias: [1, H, F]
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(1, num_heads, out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, x, edge_index, edge_type, res_attn=None):
        """
        x: [N, C_in] or ([N_src, C_src], [N_dst, C_dst])
        edge_index: [2, E]
        edge_type: [E]
        res_attn: [E, H] (Optional)
        """
        # 1. Node feature projection
        if isinstance(x, tuple):
            h_src, h_dst = self.feat_drop(x[0]), self.feat_drop(x[1])
            # [N, H * F] -> [N, H, F]
            feat_src = self.fc_src(h_src).view(-1, self.num_heads, self.out_channels)
            feat_dst = self.fc_dst(h_dst).view(-1, self.num_heads, self.out_channels)
        else:
            h_src = h_dst = self.feat_drop(x)
            # [N, H * F] -> [N, H, F]
            feat_src = feat_dst = self.fc(h_src).view(-1, self.num_heads, self.out_channels)

        # 2. Edge feature projection
        # [E] -> [E, D] -> [E, H * D] -> [E, H, D]
        e_emb = self.edge_emb(edge_type)
        e_feat = self.fc_e(e_emb).view(-1, self.num_heads, self.edge_dim)

        # 3. Message Passing (Propagate)
        # Internal: calls message -> aggregate -> update
        # out: [N, H, F]
        out = self.propagate(edge_index, x=(feat_src, feat_dst), e_feat=e_feat, res_attn=res_attn)

        # 4. Residual connection
        if self.res_fc is not None:
            # [N, C_in] -> [N, H * F] -> [N, H, F]
            resval = self.res_fc(h_dst).view(h_dst.size(0), self.num_heads, self.out_channels)
            out = out + resval

        # 5. Bias addition
        if self.bias_param is not None:
            # [N, H, F] + [1, H, F]
            out = out + self.bias_param

        # 6. Activation
        if self.activation:
            out = self.activation(out)

        # Returns [N, H, F] and detached attention weights [E, H]
        return out, self._alpha

    def message(self, x_i, x_j, e_feat, index, ptr, size_i, res_attn):
        """
        x_j (source): [E, H, F]
        x_i (target): [E, H, F]
        e_feat: [E, H, D]
        """
        # Compute attention scores: el + er + ee
        # (x_j * attn_l).sum -> [E, H, 1]
        el = (x_j * self.attn_l).sum(dim=-1, keepdim=True)
        er = (x_i * self.attn_r).sum(dim=-1, keepdim=True)
        ee = (e_feat * self.attn_e).sum(dim=-1, keepdim=True)

        # [E, H, 1] -> [E, H]
        alpha_score = self.leaky_relu(el + er + ee).squeeze(-1)

        # Softmax normalization across neighbors
        # alpha: [E, H]
        alpha = softmax(alpha_score, index, ptr, size_i)
        alpha = self.attn_drop(alpha)

        # Residual attention logic (if res_attn is provided)
        if res_attn is not None:
            alpha = alpha * (1 - self.alpha) + res_attn * self.alpha

        # Store attention for analysis
        self._alpha = alpha.detach()

        # Weighted message: [E, H, F] * [E, H, 1] -> [E, H, F]
        return x_j * alpha.unsqueeze(-1)
    

class HeCoGCNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HeCoGCNConv, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, bias=bias)
        self.act = nn.PReLU()

        nn.init.xavier_normal_(self.conv.lin.weight, gain=1.414)
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        # x: [N, in_ft], edge_index: [2, E]
        out = self.conv(x, edge_index)
        return self.act(out)