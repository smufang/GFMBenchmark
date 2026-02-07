import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
from models import DGI, GraphCL
from layers import GCN, AvgReadout
import torch_scatter
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv,GraphConv,GATConv
import time

class supervised(nn.Module):
    def __init__(self,input_dim,hid_dim,output_dim,num_layers,type):
        super(supervised,self).__init__()
        self.mlp = MLP(hid_dim,output_dim)
        self.num_layers_num=num_layers
        self.type=type
        self.g_net, self.bns = self.create_net(input_dim,hid_dim,num_layers)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                if self.type == 'gcn':
                    nn = GCNConv(hidden_dim, hidden_dim)
                elif self.type == 'gat':
                    nn = GATConv(hidden_dim, hidden_dim)
            else:
                if self.type == 'gcn':
                    nn = GCNConv(input_dim, hidden_dim)
                elif self.type == 'gat':
                    nn = GATConv(input_dim, hidden_dim)

            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self,x,edge_index,idx,batch):
        for i in range(self.num_layers_num):
            # print("i",i)
            
            if i:
                # print(graph_output.shape)
                graph_output = self.convs[i](graph_output , edge_index) + graph_output
            else:
                graph_output = self.convs[i](x , edge_index)
            rawret = self.mlp(graph_output)
            rawret = torch_scatter.scatter(src=rawret, index=batch, dim=0, reduce='mean')
            ret = rawret[idx]
        return ret

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

class ConditionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(ConditionNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_fc(x)
        for layer in self.hidden_fc:
            x = layer(x)
        output = self.output_fc(x)
        return output

class downprompt(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nb_classes, think_layer_num, condition_layer_num,type,usemlp):
        super(downprompt, self).__init__()
        self.nb_classes = nb_classes
        self.condition_layers_num = condition_layer_num
        self.think_layer_num = think_layer_num
        self.layer_norm = nn.LayerNorm(out_dim)
        self.condition_net = ConditionNet(in_dim, hid_dim, in_dim, condition_layer_num)
        self.preprompt = downstreamprompt(out_dim)
        self.graphprompt = downstreamprompt(in_dim)
        self.GPFplus = GPFplusAtt(out_dim,p_num=5)
        self.allinone = allinone(out_dim)
        self.GPF = GPF(out_dim)
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.mlp = MLP(in_dim, nb_classes)
        self.type = type
        self.usemlp = usemlp
        self.ave = None
    
    def encode(self, x, edge_index, gcn):
        embed = gcn(x, edge_index)

        if self.usemlp == 'no':
            if self.type == 'GPFplus':
                x = self.GPFplus.add(x)
                embed = gcn(x, edge_index)
            elif self.type == 'Graphprompt':
                embed = self.graphprompt(embed)
            elif self.type == 'GPF':
                x = self.GPF.add(x)
                embed = gcn(x, edge_index)
            elif self.type == 'allinone':
                promptx ,promptedge = self.allinone(x,edge_index)
                embed = gcn(promptx, promptedge)
                embed = embed[100:]
            
        elif self.usemlp == 'yes':
            embed = self.mlp(embed)
        
        return embed
    
    def support(self, x, edge_index, gcn, idx, batch, labels, task_name):
        embed = self.encode(x, edge_index, gcn)
        if task_name == "graph":
            embed = torch_scatter.scatter(src=embed, index=batch, dim=0, reduce='mean')
            rawret = embed[idx]
        elif task_name == "edge":
            rawret = torch.cat([embed[idx[0]], embed[idx[1]]], dim=-1)
        else:
             rawret = embed[idx]

        self.ave = averageemb(labels.long(), rawret, self.nb_classes)
    
    @torch.no_grad()
    def predict(self, x, edge_index, gcn, idx, batch, task_name):
        embed = self.encode(x, edge_index, gcn)
        if task_name == "graph":
            g_embed = torch_scatter.scatter(src=embed, index=batch, dim=0, reduce='mean')
            rawret = g_embed[idx]
        elif task_name == "edge":
            rawret = torch.cat([embed[idx[0]], embed[idx[1]]], dim=-1)
        else:
            rawret = embed[idx]
        
        num = rawret.size(0)
        rawret = torch.cat([rawret, self.ave], dim=0)
        rawret = F.normalize(rawret, dim=-1)
        sim = rawret @ rawret.t()
        logits = sim[:num, num:]
        
        return logits.argmax(dim=1)

    def mlp_forward(self, x, edge_index, gcn, idx, batch, task_name):
        embed = gcn(x, edge_index)

        if task_name == "graph":
            embed = torch_scatter.scatter(embed, batch, dim=0, reduce='mean')
            logits = self.mlp(embed)[idx]
        elif task_name == "edge":
            embed = embed[idx[0]] + embed[idx[1]]
            logits = self.mlp(embed)
        else:
            logits = self.mlp(embed)[idx]

        return logits.argmax(dim=1)

    
    def forward(self, x, edge_index, gcn, idx, batch, labels=None, train=0):
        if train == 0:
            start_time = time.perf_counter()  

        if self.usemlp == 'no':
            embed = gcn(x, edge_index)
            if self.type == 'GPFplus':
                # print('useGPFplus')
                x = self.GPFplus.add(x)
                embed = gcn(x, edge_index)
            elif self.type == 'Graphprompt':
                # print('useGraphprompt')
                embed = self.graphprompt(embed)
            elif self.type == 'GPF':
                x = self.GPF.add(x)
                embed = gcn(x, edge_index)
            elif self.type == 'allinone':
                promptx ,promptedge = self.allinone(x,edge_index)
                embed = gcn(promptx, promptedge)
                embed = embed[100:]
                
            rawret = torch_scatter.scatter(src=embed, index=batch, dim=0, reduce='mean')
            rawret = rawret[idx]

            if train == 1:
                self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)

            num = rawret.shape[0]
            ret = torch.zeros(num, self.nb_classes, device=rawret.device)  # Ensure device match
            rawret = torch.cat((rawret, self.ave), dim=0)
            rawret = F.normalize(rawret, dim=-1)  # Normalize for cosine similarity stability
            similarity = torch.mm(rawret, rawret.t())  # Compute pairwise similarity
            ret = similarity[:num, num:]
            ret = F.softmax(ret, dim=1)
        
        elif self.usemlp == 'yes':
            embed = gcn(x, edge_index)
            rawret = self.mlp(embed)
            rawret = torch_scatter.scatter(src=rawret, index=batch, dim=0, reduce='mean')
            ret = rawret[idx]
        if train == 0:
            now_time = time.perf_counter()
            return ret , (now_time - start_time) * 1000
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, graph_embedding):
        graph_embedding = self.weight * graph_embedding
        return graph_embedding

def averageemb(labels, rawret, nb_class):
    return torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

class GPF(nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x):

        ret = x + self.global_emb

        return ret


class allinone(nn.Module):
    def __init__(self,token_dim):
        super(allinone,self).__init__()
        self.inner_prune = 0.3
        self.cross_prune = 0.1
        self.token = torch.nn.Parameter(torch.FloatTensor(100, token_dim))
        self.token_init(init_method="kaiming_uniform")
    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def forward(self,x,edge_index):
        token_dot = torch.mm(self.token, torch.transpose(self.token, 0, 1))

        token_sim = torch.sigmoid(token_dot)  # 0-1
        # print(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, torch.zeros_like(token_sim), token_sim)
        inner_edge_index = inner_adj.nonzero().t().contiguous()
            

        token_num = self.token.shape[0]
        g_edge_index = edge_index + token_num
        
        cross_dot = torch.mm(x, torch.transpose(self.token, 0, 1))
        cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
        cross_adj = torch.where(cross_sim < self.cross_prune, torch.zeros_like(cross_sim), cross_sim)
        
        cross_edge_index = cross_adj.nonzero().t().contiguous()
        cross_edge_index[1] = cross_edge_index[1] + token_num
        
        x = torch.cat([x,self.token], dim=0)
        edge_index = torch.cat([g_edge_index, inner_edge_index,cross_edge_index], dim=1)

        return x,edge_index
