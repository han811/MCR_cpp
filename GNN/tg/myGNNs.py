import pickle

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, Linear, BatchNorm

class myGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu', is_save_hiddens=False):
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

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0

    def forward(self, x, edge_index):
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)
            
            x = self.conv2(x, edge_index)
            x = self.batchnorm2(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = self.Linear(x)
            x = self.activation(x)
            hidden_state.append(x)

            x = self.sigmoid(x)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
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

            x = self.sigmoid(x)
        return x


class myGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu', is_save_hiddens=False):
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

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0

    def forward(self, x, edge_index):
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)
            
            x = self.conv2(x, edge_index)
            x = self.batchnorm2(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = self.Linear(x)
            x = self.activation(x)
            hidden_state.append(x)
            
            x = self.sigmoid(x)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
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
            
            x = self.sigmoid(x)
        return x


class myGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2, activation='relu', is_save_hiddens=False):
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

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0

    def forward(self, x, edge_index):
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)
            
            x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = self.batchnorm2(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = self.Linear(x)
            x = self.activation(x)
            hidden_state.append(x)

            x = self.sigmoid(x)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
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

            x = self.sigmoid(x)
        return x, (attention1, attention2)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_dim, activation='relu', dropout=0.35, is_save_hiddens=False):
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

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0
    
    def forward(self, x, edge_index, l):
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = torch.cat([x,l.unsqueeze(1)],dim=-1)

            x = self.conv2(x, edge_index)
            x = self.batchnorm2(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = self.Linear(x)
            hidden_state.append(x)

            x = x.mean(dim=0)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
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

            x = x.mean(dim=0)
        return x

class GATDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_dim, activation='relu', dropout=0.35, is_save_hiddens=False):
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

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0
    
    def forward(self, x, edge_index, z):
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = torch.cat([x,z.repeat((25,1))],dim=-1)

            x = self.conv2(x,edge_index)
            x = self.batchnorm2(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            x = self.Linear(x)
            hidden_state.append(x)

            x = self.sigmoid(x)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            x = self.dropout(x)

            x = torch.cat([x,z.repeat((25,1))],dim=-1)

            x = self.conv2(x,edge_index)
            x = self.batchnorm2(x)
            x = self.activation(x)
            x = self.dropout(x)

            x = self.Linear(x)

            x = self.sigmoid(x)

        return x


class myGATcVAE(nn.Module):
    def __init__(self,
     en_in_channels, en_hidden_channels,
     de_in_channels, de_hidden_channels,
     z_dim, activation='relu', dropout=0.35, is_save_hiddens=False):
        super(myGATcVAE, self).__init__()

        self.encoder = GATEncoder(en_in_channels, en_hidden_channels, z_dim, activation, dropout, is_save_hiddens)
        self.decoder = GATDecoder(de_in_channels, de_hidden_channels, z_dim, activation, dropout, is_save_hiddens)
        self.z_dim = z_dim

    def reparameterization(self, mu, log_var):
        eps = torch.randn(1, self.z_dim).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, edge_index, l):
        z = self.encoder(x, edge_index, l)
        z_splits = z.split(self.z_dim,dim=0)
        z_mu = z_splits[0]
        z_log_var = z_splits[1]
        z = self.reparameterization(z_mu, z_log_var)
        l = self.decoder(x, edge_index, z)
        return l, z_mu, z_log_var


#################################
### Embedding sub-graph parts ###
#################################


class myGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation='relu'):
        super(myGraphEncoder, self).__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv1.reset_parameters()
        self.Linear1 = Linear(hidden_channels, out_channels)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.Linear1(x)
        x = x.mean(dim=0)
        return x

class myGraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation='relu'):
        super(myGraphDecoder, self).__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv1.reset_parameters()
        self.Linear1 = Linear(hidden_channels, out_channels)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.Linear1(x)
        return x

class myGraphAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, activation='relu', save_name=None):
        super(myGraphAE, self).__init__()
        self.enc = myGraphEncoder(in_channels, hidden_channels, latent_channels, activation=activation)
        self.dec = myGraphDecoder(latent_channels, hidden_channels, in_channels, activation=activation)
    
    def forward(self, x, edge_index):
        n_nodes = x.size()[0]
        x = self.enc(x, edge_index)
        x = x.repeat(n_nodes, 1)
        x = self.dec(x, edge_index)
        return x