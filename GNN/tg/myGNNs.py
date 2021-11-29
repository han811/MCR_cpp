import pickle

import torch
import torch.nn as nn
from torch.serialization import save

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
    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.2, activation='relu', is_save_hiddens=False):
        super(myGraphSAGE, self).__init__()

        self.convs = nn.ModuleList()
        if num_layers>=2:
            pre_channel = in_channels
            for hidden_channel in hidden_channels:
                self.convs.append(SAGEConv(pre_channel,hidden_channel))
                pre_channel = hidden_channel
            self.Linear = Linear(hidden_channels[-1],1)
        else:
            self.convs.append(SAGEConv(in_channels,hidden_channels))
            self.Linear = Linear(hidden_channels,1)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0

    def forward(self, x, edge_index):
<<<<<<< HEAD
        for layer in self.convs:
            x = layer(x, edge_index)
=======
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            hidden_state.append(x)
            
            x = self.conv2(x, edge_index)
            x = self.activation(x)
            hidden_state.append(x)

            x = self.Linear(x)
            hidden_state.append(x)
            
            x = self.sigmoid(x)
            hidden_state.append(x)

            if self.num % 1000 == 0:
                with open(f'./hidden_features/{self.__class__.__name__}/hidden_features_{self.num2}', 'wb') as f:
                    pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
                self.num2 += 1
                self.num = 0
            self.num += 1
        else:
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            
            x = self.conv2(x, edge_index)
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
            x = self.activation(x)
        x = self.Linear(x)
        x = self.sigmoid(x)
        return x


class myGraphSAGE_ED(nn.Module):
    def __init__(self, in_channels, embedding_channels, hidden_channels, dropout=0.2, activation='relu', is_save_hiddens=False):
        super(myGraphSAGE_ED, self).__init__()
        
        self.embedding = Linear(in_channels, embedding_channels)
        self.embedding.reset_parameters()
        self.conv1 = SAGEConv(embedding_channels,hidden_channels)
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
            x = self.embedding(x)

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
            # x = self.batchnorm1(x)
            x = self.activation(x)
            # x = self.dropout(x)
            
            x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)
            # x = self.batchnorm2(x)
            x = self.activation(x)
            # x = self.dropout(x)

            x = self.Linear(x)
            x = self.activation(x)

            x = self.sigmoid(x)
        return x, (attention1, attention2)







'''
    Graph conditional VAE to resolve noisy label problem
'''
class SAGEGraphEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_channels=64, activation='relu', dropout=0.2):
        super(SAGEGraphEmbedding, self).__init__()

<<<<<<< HEAD
        self.conv = SAGEConv(in_channels,embedding_channels)
        self.conv.reset_parameters()
=======
        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv2 = SAGEConv(hidden_channels,embedding_channels)
        
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
<<<<<<< HEAD
        x = self.conv(x,edge_index)
=======
        x = self.conv1(x,edge_index)
        x = self.activation(x)
        x = self.conv2(x,edge_index)
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_dim, activation='relu', dropout=0.2, is_save_hiddens=False):
        super(SAGEEncoder, self).__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.linear1 = Linear(hidden_channels,2*z_dim)
 
        self.conv1.reset_parameters()

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.is_save_hiddens = is_save_hiddens
        if self.is_save_hiddens:
            self.num = 0
            self.num2 = 0
    
    def forward(self, x, edge_index, batch_size):
<<<<<<< HEAD
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.linear1(x)
        x = torch.reshape(x,(batch_size,-1,x.size()[1]))
        x = x.mean(dim=1)
=======
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            # x = torch.cat([x,l.unsqueeze(1)],dim=-1)

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
            x = self.activation(x)
            x = self.linear1(x)
            x = torch.reshape(x,(batch_size,-1,x.size()[1]))
            x = x.mean(dim=1)
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
        return x

class SAGEDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation='relu', dropout=0.2, is_save_hiddens=False):
        super(SAGEDecoder, self).__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv2 = SAGEConv(hidden_channels,1)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

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
<<<<<<< HEAD
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x,edge_index)
        x = self.sigmoid(x)
=======
        if self.is_save_hiddens:
            hidden_state = []
            hidden_state.append(x)
            x = self.conv1(x, edge_index)
            x = self.batchnorm1(x)
            x = self.activation(x)
            hidden_state.append(x)
            x = self.dropout(x)

            # x = torch.cat([x,z.repeat((25,1))],dim=-1)

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
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.sigmoid(x)
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
        return x

class mySAGEcVAE(nn.Module):
    def __init__(self,
     embedding_in_channels, embedding_hidden_channels, embedding_channels,
     en_hidden_channels, de_hidden_channels,
     z_dim, activation='relu', dropout=0.2, is_save_hiddens=False):
        super(mySAGEcVAE, self).__init__()

        self.embedding = SAGEGraphEmbedding(embedding_in_channels, embedding_hidden_channels, embedding_channels, activation=activation, dropout=dropout)

        self.encoder = SAGEEncoder(embedding_channels+1, en_hidden_channels, z_dim, activation=activation, dropout=dropout, is_save_hiddens=is_save_hiddens)
        self.decoder = SAGEDecoder(embedding_channels+z_dim, de_hidden_channels, activation=activation, dropout=dropout, is_save_hiddens=is_save_hiddens)
        self.z_dim = z_dim

    def reparameterization(self, mu, log_var, batch_size):
        eps = torch.randn(batch_size, self.z_dim).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, edge_index, l, batch_size):
        c = self.embedding(x, edge_index)
        z = torch.cat([l.unsqueeze(-1),c],dim=1)
<<<<<<< HEAD
        # print(z.size())
        # print(c.size())
        z = self.encoder(z, edge_index, batch_size)
        z_mu, z_log_var = z.split(self.z_dim,dim=1)
        z = self.reparameterization(z_mu, z_log_var,batch_size)
=======
        z = self.encoder(z, edge_index, batch_size)
        z_mu, z_log_var = z.split(self.z_dim,dim=1)
        z = self.reparameterization(z_mu, z_log_var)
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
        z_next = []
        repeat_num = int(c.size()[0]/batch_size)
        for tmp_idx, tmp_z in enumerate(z):
            if tmp_idx==0:
                z_next = tmp_z.unsqueeze(0)
                z_next = z_next.repeat(repeat_num,1)
            else:
                tmp_z_next = tmp_z.unsqueeze(0)
                tmp_z_next = tmp_z_next.repeat(repeat_num,1)
                z_next = torch.cat([z_next,tmp_z_next], dim=0)
        z = torch.cat([z_next,c],dim=1)
        l = self.decoder(z, edge_index)
<<<<<<< HEAD
        return l, z_mu, z_log_var
=======
        return l, z_mu.clone().detach(), z_log_var.clone().detach()






>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e









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