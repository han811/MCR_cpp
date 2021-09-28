import os
import pickle

import torch
import torch.nn as nn
from torch.nn.modules import activation
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, Linear, BatchNorm
from torch_geometric.loader import DataLoader

from tg_Preprocessing import load_tg_data

class myGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu'):
        super(myGCN, self).__init__()

        self.conv1 = GCNConv(in_channels,hidden_channels)
        self.conv1.reset_parameters()
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm1.reset_parameters()
        self.conv2 = GCNConv(hidden_channels,hidden_channels)
        self.conv2.reset_parameters()
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.batchnorm2.reset_parameters()
        self.Linear = Linear(hidden_channels,1)
        self.Linear.reset_parameters()

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.Linear(x)
        x = self.activation(x)

        return self.sigmoid(x)


class myGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu'):
        super(myGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv1.reset_parameters()
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm1.reset_parameters()
        self.conv2 = SAGEConv(hidden_channels,hidden_channels)
        self.conv2.reset_parameters()
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.batchnorm2.reset_parameters()
        self.Linear = Linear(hidden_channels,1)
        self.Linear.reset_parameters()

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.Linear(x)
        x = self.activation(x)

        return self.sigmoid(x)


class myGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu'):
        super(myGAT, self).__init__()

        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv1.reset_parameters()
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm1.reset_parameters()
        self.conv2 = GATv2Conv(hidden_channels,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv2.reset_parameters()
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.batchnorm2.reset_parameters()
        self.Linear = Linear(hidden_channels,1)
        self.Linear.reset_parameters()

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.Linear(x)
        x = self.activation(x)

        return self.sigmoid(x), (attention1, attention2)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_dim, activation='relu', dropout=0.35):
        super(GATEncoder, self).__init__()

        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv1.reset_parameters()
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm1.reset_parameters()

        self.conv2 = GATv2Conv(hidden_channels+1,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv2.reset_parameters()
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.batchnorm2.reset_parameters()

        self.Linear = Linear(hidden_channels, 2*z_dim)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, l):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.cat([x,l.unsqueeze(1)],dim=-1)

        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.Linear(x)

        return x.mean(dim=0)

class GATDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_dim, activation='relu', dropout=0.35):
        super(GATDecoder, self).__init__()

        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv1.reset_parameters()
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm1.reset_parameters()

        self.conv2 = GATv2Conv(hidden_channels+z_dim,hidden_channels,heads=4,concat=False,dropout=dropout)
        self.conv2.reset_parameters()
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.batchnorm2.reset_parameters()

        self.Linear = Linear(hidden_channels, 1)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index, z):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.cat([x,z.repeat((25,1))],dim=-1)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.Linear(x)

        x = self.sigmoid(x)

        return x


class GraphcVAE(nn.Module):
    def __init__(self,
     en_in_channels, en_hidden_channels,
     de_in_channels, de_hidden_channels,
     z_dim, activation='relu', dropout=0.35):
        super(GraphcVAE, self).__init__()

        self.encoder = GATEncoder(en_in_channels, en_hidden_channels, z_dim, activation, dropout)
        self.decoder = GATDecoder(en_in_channels, en_hidden_channels, z_dim, activation, dropout)
        self.z_dim = z_dim

    def reparameterization(self, mu, log_var):
        eps = torch.randn(1, self.z_dim).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, edge_index, l):
        z = self.encoder(x, edge_index, l)
        z_splits = z.split(2,dim=0)
        z_mu = z_splits[0]
        z_log_var = z_splits[1]
        z = self.reparameterization(z_mu, z_log_var)
        l = self.decoder(x, edge_index, z)
        return l, z_mu, z_log_var