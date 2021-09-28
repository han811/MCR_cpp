import os
import pickle

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, Linear, BatchNorm, GCN, GraphSAGE
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.norm import batch_norm


''' linear layer generator '''
def layer_generator(in_f, out_f, hidden_layers = (128, 256, 128)):
    n_hidden_layers = len(hidden_layers)
    return nn.ModuleList([Linear(in_f, hidden_layers[i],weight_initializer='uniform') if i == 0 else
                            Linear(hidden_layers[i-1], out_f,weight_initializer='uniform') if i == n_hidden_layers else
                            Linear(hidden_layers[i-1], hidden_layers[i],weight_initializer='uniform') for i in range(n_hidden_layers+1)]), n_hidden_layers


''' GNN node classification model structure '''
class myGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, enc_channels, hidden_channels, num_layers, out_channels, dropout=0.0, act=nn.ReLU(), clf_num_layers=4):
        super(myGraphSAGE, self).__init__()
        self.enc = nn.Linear(in_channels,enc_channels)
        torch.nn.init.xavier_uniform(self.enc.weight)
        self.enc_batchnorm = BatchNorm(enc_channels)

        self.GraphSAGE = GraphSAGE(in_channels=enc_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, dropout=dropout, act=act, jk='cat')
        
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.activation = act
        for _ in range(clf_num_layers):
            self.layers.append(Linear(out_channels,out_channels))
            self.batchnorm_layers.append(BatchNorm(out_channels))
        self.layers.append(Linear(out_channels,1))
        for layer in self.layers:
            torch.nn.init.xavier_uniform(layer.weight)

        self.num = 0
        self.num2 = 0
    def forward(self, x, edge_index):
        hidden_state = []
        x = self.enc(x)
        x = self.enc_batchnorm(x)
        hidden_state.append(x)
        x = self.GraphSAGE(x, edge_index)
        hidden_state.append(x)
        if len(self.layers)>1:
            for idx,layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = self.batchnorm_layers[idx](x)
                x = self.activation(x)
                hidden_state.append(x)
        x = self.layers[-1](x)
        hidden_state.append(x)
        x = self.sigmoid(x)
        hidden_state.append(x)
        if self.num % 1000 == 0:
            with open(f'./hidden_features/hidden_features_{self.num2}', 'wb') as f:
                pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
            self.num2 += 1
            self.num = 0
        self.num += 1
        return x



class myGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.0, act=nn.ReLU()):
        super(myGCN, self).__init__()
        self.GCN = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=hidden_channels, dropout=dropout, act=act)
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList()
        self.activation = nn.ELU()
        for _ in range(3):
            self.layers.append(Linear(hidden_channels,hidden_channels))
        self.layers.append(Linear(hidden_channels,1))
        self.num = 0
        self.num2 = 0
    def forward(self, x, edge_index):
        # hidden_state = []
        x = self.GCN(x, edge_index)
        # hidden_state.append(x)
        for layer in self.layers:
            x = self.activation(layer(x))
            # hidden_state.append(x)
        x = self.sigmoid(x)
        # hidden_state.append(x)
        # if self.num % 1000 == 0:
        #     with open(f'./hidden_features/hidden_features_{self.num2}', 'wb') as f:
        #         pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
        #     self.num2 += 1
        #     self.num = 0
        # self.num += 1
        return x

class GATclf(torch.nn.Module):
    def __init__(self, n_node_feature, n_encoding_feature,
     n_mid_features = (64, 128, 64),
     encoder_hidden_layers = (128, 256, 128),
     node_hidden_layers = ((128, 256, 128),
                            (128, 256, 128),
                            (128, 256, 128)),
     output_hidden_layers = (128, 256, 128),
     message_passing_steps=3, activation=None):
        super(GATclf, self).__init__()
        torch.manual_seed(811)

        self.n_node_feature = n_node_feature
        self.n_encoding_feature = n_encoding_feature
        self.n_mid_features = n_mid_features
        
        self.message_passing_steps = message_passing_steps
        
        self.encoder_hidden_layers = encoder_hidden_layers
        self.node_hidden_layers = node_hidden_layers
        self.output_hidden_layers = output_hidden_layers

        self.encoder_layer = nn.ModuleList()
        self.GAT_layer = nn.ModuleList()
        self.GAT_batchnorm = nn.ModuleList()
        self.node_update_layer = nn.ModuleList()

        self.encoder_layer, _ = layer_generator(self.n_node_feature,self.n_encoding_feature,self.encoder_hidden_layers)

        for step in range(self.message_passing_steps):
            if step==0:
                # GAT_layer = GCNConv(self.n_encoding_feature, self.n_mid_features[step])
                GAT_layer = GATConv(self.n_encoding_feature, self.n_mid_features[step], heads=1, concat=True, negative_slope=0.2)
                self.GAT_layer.append(GAT_layer)
                batchnorm = BatchNorm(self.n_mid_features[step])
                self.GAT_batchnorm.append(batchnorm)
            else:
                node_update_layer, _ = layer_generator(self.n_mid_features[step-1], self.n_mid_features[step-1], self.node_hidden_layers[step-1])
                self.node_update_layer.append(node_update_layer)

                # GAT_layer = GCNConv(self.n_mid_features[step-1], self.n_mid_features[step])
                GAT_layer = GATConv(self.n_mid_features[step-1], self.n_mid_features[step], heads=1, concat=True, negative_slope=0.2)
                self.GAT_layer.append(GAT_layer)
                batchnorm = BatchNorm(self.n_mid_features[step])
                self.GAT_batchnorm.append(batchnorm)

        self.output_update_layer, _ = layer_generator(self.n_mid_features[-1],1,self.output_hidden_layers)

        self.dropout = nn.Dropout(0.35)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        else:
            self.activation = None

        self.sigmoid = nn.Sigmoid()

        self.num = 0
        self.num2 = 0

    def forward(self, x, edge_index):
        hidden_state = []
        for layer in self.encoder_layer[:-1]:
            x = self.activation(layer(x))
            hidden_state.append(x.clone().detach().tolist())
            x = self.dropout(x)
        x = self.encoder_layer[-1](x)
        hidden_state.append(x.clone().detach().tolist())


        for step in range(self.message_passing_steps-1):
            x = self.GAT_layer[step](x,edge_index)
            x = self.GAT_batchnorm[step]
            x = self.activation(x)
            hidden_state.append(x.clone().detach().tolist())
            x = self.dropout(x)

            for layer in self.node_update_layer[step]:
                x = self.activation(layer(x))
                hidden_state.append(x.clone().detach().tolist())
                x = self.dropout(x)
        x = self.GAT_layer[-1](x,edge_index)
        x = self.GAT_batchnorm[-1](x)
        x = self.activation(x)
        # x = self.activation(self.GAT_layer[-1](x,edge_index))
        hidden_state.append(x.clone().detach().tolist())

        for layer in self.output_update_layer[:-1]:
            x = self.activation(layer(x))
            hidden_state.append(x.clone().detach().tolist())
            x = self.dropout(x)
        x = self.output_update_layer[-1](x)
        hidden_state.append(x.clone().detach().tolist())

        x = self.sigmoid(x)
        hidden_state.append(x.clone().detach().tolist())
        if self.num % 3431 == 0:
            with open(f'./hidden_features/hidden_features_{self.num2}', 'wb') as f:
                pickle.dump(hidden_state,f,pickle.HIGHEST_PROTOCOL)
            self.num2 += 1
            self.num = 0
        self.num += 1

        return x