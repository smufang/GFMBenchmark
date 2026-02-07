import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import GCN, AvgReadout
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_min 
import torch_sparse
import torch_scatter
import tqdm
import numpy as np
from sklearn.decomposition import PCA
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast,SingleBranchContrast
from torch_geometric.utils import negative_sampling

def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
    return res

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)

class PrePrompt(nn.Module):
    def __init__(self,n_in,n_h, num_layers_num, dropout,sample=None,premodel = 'LP'):
        super(PrePrompt, self).__init__()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        # print(self.gcn)
        self.aug1 = A.Identity()
        self.aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                                A.NodeDropping(pn=0.1),
                                A.FeatureMasking(pf=0.1),
                                A.EdgeRemoving(pe=0.1)], 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G')
        self.DGIloss = SingleBranchContrast(loss=L.JSD(), mode='G2L')
        self.premodel = premodel
        self.project = torch.nn.Linear(n_h, n_h)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    
    def forward(self, x, edge_index):
        if self.premodel == 'LP':
            g = self.gcn(x, edge_index)
            negative_sample = torch.tensor(sample, dtype=torch.int64).cuda()
            loss = compareloss(g, negative_sample, temperature=1)
        if self.premodel == 'GraphCL':
            x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
            x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
            # print(x1.shape)
            g1 = self.gcn(x1, edge_index1)
            g2 = self.gcn(x2, edge_index2)         
            loss = self.contrast_model(g1=g1, g2=g2)  
        if self.premodel == 'DGI':
            z = self.gcn(x, edge_index)
            g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
            zn = self.gcn(*self.corruption(x, edge_index))
            loss = self.DGIloss(h=z, g=g, hn=zn)  
        return loss

    def embed(self, x, edge_index):
        g = self.gcn(x, edge_index)
        return g


class textprompt(nn.Module):
    def __init__(self,hid_units,type_):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding


def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(edge_index,n):
    nodenum = edge_index.max().item() + 1  # Number of nodes, assuming 0-based indexing

    # Convert edge_index to adjacency list format
    adj_dict = {i: set() for i in range(nodenum)}
    for i, j in edge_index.t().tolist():
        adj_dict[i].add(j)
        adj_dict[j].add(i)  # Since undirected graph, add both directions

    # Prepare the result array
    res = np.zeros((nodenum, 1 + n), dtype=int)  # First column for positive sample, rest for negative samples
    whole = np.array(range(nodenum))

    # Iterate through each node to sample negative samples
    for i in range(nodenum):
        # Get the list of neighbors (positive edges)
        neighbors = list(adj_dict[i])

        # Get the list of non-neighbors (potential negative samples)
        non_neighbors = np.setdiff1d(whole, neighbors)

        # If there are no neighbors, just take the node itself (this is a rare case, as most nodes should have neighbors)
        if len(neighbors) == 0:
            res[i][0] = i
        else:
            # Select the first neighbor as a positive sample
            res[i][0] = neighbors[0]

        # Select n random negative samples
        np.random.shuffle(non_neighbors)
        res[i][1:1 + n] = non_neighbors[:n]

    return res