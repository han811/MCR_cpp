import os
import pickle

import torch
import torch.nn as nn
from torch.nn.modules import activation
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv, GATConv, Linear
from torch_geometric.nn import BatchNorm, GCN, GraphSAGE, GATConv
from torch_geometric.loader import DataLoader

from tg_Preprocessing import load_tg_data


''' linear layer generator '''
def layer_generator(in_f, out_f, hidden_layers = (128, 256, 128)):
    n_hidden_layers = len(hidden_layers)
    return nn.ModuleList([Linear(in_f, hidden_layers[i]) if i == 0 else
                            Linear(hidden_layers[i-1], out_f) if i == n_hidden_layers else
                            Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(n_hidden_layers+1)]), n_hidden_layers

class Encoder(nn.Module):
    def __init__(self,
     graph_in_channels, graph_hidden_channels, graph_num_layers, graph_out_channels,
     hidden_channels, num_layers, z_dim,
     dropout=0.0, act=nn.ReLU(), graph_linear_num_layers=4):
        super(Encoder, self).__init__()
        self.GraphEncoding = GraphSAGE(graph_in_channels, graph_hidden_channels, graph_num_layers, graph_hidden_channels, dropout, act)
        self.LinearEncoding = nn.ModuleList()
        for _ in range(graph_linear_num_layers):
            layer = Linear(graph_hidden_channels, graph_hidden_channels)
            self.LinearEncoding.append(layer)
        layer = Linear(graph_hidden_channels, graph_out_channels)
        self.LinearEncoding.append(layer)
        self.activation = nn.ELU()

        self.TotalLinearEncoding = nn.ModuleList()
        layer = Linear(graph_out_channels+1,hidden_channels)
        self.TotalLinearEncoding.append(layer)
        for _ in range(num_layers):
            layer = Linear(hidden_channels,hidden_channels)
            self.TotalLinearEncoding.append(layer)
        layer = Linear(hidden_channels,z_dim)
        self.TotalLinearEncoding.append(layer)
    
    def forward(self, x, edge_index, l):
        x = self.GraphEncoding(x,edge_index)
        for layer in self.LinearEncoding:
            x = self.activation(layer(x))
        x = torch.cat([x,l.unsqueeze(1)],dim=-1)
        for layer in self.TotalLinearEncoding:
            x = self.activation(layer(x))
        x = x.mean(dim=0)
        return x

class Decoder(nn.Module):
    def __init__(self,
     graph_in_channels, graph_hidden_channels, graph_num_layers, graph_out_channels,
     z_dim, num_z_layers,
     clf_num_layers,
     dropout=0.0, act=nn.ReLU(), graph_linear_num_layers=4):
        super(Decoder, self).__init__()
        self.z_layers = nn.ModuleList()
        for _ in range(num_z_layers):
            layer = nn.Linear(z_dim,z_dim)
            self.z_layers.append(layer)
        self.GraphEncoding = GraphSAGE(graph_in_channels, graph_hidden_channels, graph_num_layers, graph_out_channels, dropout, act)
        self.clf_layers = nn.ModuleList()

        layer = Linear(graph_out_channels+z_dim, graph_out_channels)
        self.clf_layers.append(layer)
        for _ in range(clf_num_layers):
            layer = Linear(graph_out_channels, graph_out_channels)
            self.clf_layers.append(layer)
        layer = Linear(graph_out_channels, 1)
        self.clf_layers.append(layer)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ELU()
    
    def forward(self, x, edge_index, z):
        for layer in self.z_layers:
            z = self.activation(layer(z))
        x = self.GraphEncoding(x, edge_index)
        x = torch.cat([x,z.repeat((25,1))],dim=1)
        for layer in self.clf_layers:
            x = self.activation(layer(x))
        return self.sigmoid(x)

class GraphcVAE(nn.Module):
    def __init__(self,
     en_graph_in_channels, en_graph_hidden_channels, en_graph_num_layers, en_graph_out_channels,
     hidden_channels, num_layers, z_dim,
     de_graph_in_channels, de_graph_hidden_channels, de_graph_num_layers, de_graph_out_channels,
     num_z_layers,
     clf_num_layers,
     dropout=0.0, act=nn.ReLU(), graph_linear_num_layers=4):
        super(GraphcVAE, self).__init__()

        self.encoder = Encoder(en_graph_in_channels, en_graph_hidden_channels, en_graph_num_layers, en_graph_out_channels, hidden_channels, num_layers, z_dim*2, dropout, act, graph_linear_num_layers)
        self.decoder = Decoder(de_graph_in_channels, de_graph_hidden_channels, de_graph_num_layers, de_graph_out_channels, z_dim, num_z_layers, clf_num_layers, dropout, act, graph_linear_num_layers)
        self.z_dim = z_dim

    def reparameterization(self, mu, log_var):
        eps = torch.randn(1, self.z_dim).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, edge_index, l):
        z = self.encoder(x, edge_index, l)
        z_mu = z[:int(z.size()[0]/2)]
        z_log_var = z[int(z.size()[0]/2):]
        z = self.reparameterization(z_mu, z_log_var)
        l = self.decoder(x, edge_index, z)
        return l, z_mu, z_log_var
        

if __name__=='__main__':
    train_set, _, _ = load_tg_data(num=0)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    GNN_cVAE = GraphcVAE(
     en_graph_in_channels=17338, en_graph_hidden_channels=256, en_graph_num_layers=1, en_graph_out_channels=128,
     hidden_channels=128, num_layers=1, z_dim=32,
     de_graph_in_channels=17338, de_graph_hidden_channels=256, de_graph_num_layers=1, de_graph_out_channels=256,
     num_z_layers=2,
     clf_num_layers=3,
     dropout=0.0, act=nn.ReLU(), graph_linear_num_layers=4)

    optimizer = optim.Adam(GNN_cVAE.parameters(),lr=1e-3, betas=(0.5, 0.999), weight_decay=5e-4)

    mse_loss = nn.MSELoss()

    for data in train_loader:
        optimizer.zero_grad()
        l, z_mu, z_log_var = GNN_cVAE(data.x,data.edge_index,data.y)
        kl = -0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim = 0)
        mse = mse_loss(l, data.y.unsqueeze(-1)) * 10
        loss = kl + mse
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(l.view(-1))
        print(data.y.view(-1))
