from math import log
from typing import List, Dict, Set, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from graph import Graph
from utils import init_weights

''' linear layer generator '''
def layer_generator(in_f, out_f, hidden_layers = (128, 256, 128)):
    n_hidden_layers = len(hidden_layers)
    return nn.ModuleList([nn.Linear(in_f, hidden_layers[i]) if i == 0 else
                            nn.Linear(hidden_layers[i-1], out_f) if i == n_hidden_layers else
                            nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(n_hidden_layers+1)]), n_hidden_layers

''' GNN node classification model structure '''
#                     Hidden(t)   Hidden(t+1)
#                        |            ^
#           *---------*  |  *------*  |  *---------*
#           |         |  |  |      |  |  |         |
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
class GNN_clf(nn.Module):
    def __init__(self, n_node_feature, n_encoding_feature, n_mid_feature,
     encoder_hidden_layers = (128, 256, 128),
     node_hidden_layers = (128, 256, 128),
     edge_hidden_layers = (128, 256, 128),
     node_hidden_layers2 = (128, 256, 128),
     output_hidden_layers = (128, 256, 128),
     message_passing_steps=3, activation=None):
        super(GNN_clf, self).__init__()

        self.n_node_feature = n_node_feature
        self.n_encoding_feature = n_encoding_feature
        self.n_mid_feature = n_mid_feature
        
        self.message_passing_steps = message_passing_steps

        self.encoder_hidden_layers = encoder_hidden_layers
        self.node_hidden_layers = node_hidden_layers
        self.edge_hidden_layers = edge_hidden_layers
        self.node_hidden_layers2 = node_hidden_layers2
        self.output_hidden_layers = output_hidden_layers


        self.encoder_layer = nn.ModuleList()
        self.node_update_layer = nn.ModuleList()
        self.edge_update_layer = nn.ModuleList()
        self.node_update_layers2 = nn.ModuleList()

        self.encoder_layer, _ = layer_generator(self.n_node_feature,self.n_encoding_feature,self.encoder_hidden_layers)
        for _ in range(self.message_passing_steps):
            node_update_layer, _ = layer_generator(self.n_encoding_feature,self.n_mid_feature,self.node_hidden_layers)
            self.node_update_layer.append(node_update_layer)
            edge_update_layer, _ = layer_generator(self.n_encoding_feature,self.n_mid_feature,self.edge_hidden_layers)
            self.edge_update_layer.append(edge_update_layer)
            node_update_layers2, _ = layer_generator(self.n_mid_feature,self.n_encoding_feature,self.node_hidden_layers2)
            self.node_update_layers2.append(node_update_layers2)
            
        self.output_update_layer, _ = layer_generator(self.n_encoding_feature,1,self.output_hidden_layers)


        # self.encoder_layer, _ = layer_generator(self.n_node_feature,self.n_encoding_feature,self.encoder_hidden_layers)
        # self.node_update_layer, _ = layer_generator(self.n_encoding_feature,self.n_mid_feature,self.node_hidden_layers)
        # self.edge_update_layer, _ = layer_generator(self.n_encoding_feature,self.n_mid_feature,self.edge_hidden_layers)
        # self.node_update_layers2, _ = layer_generator(self.n_mid_feature,self.n_encoding_feature,self.node_hidden_layers2)
        # self.output_update_layer, _ = layer_generator(self.n_encoding_feature,1,self.output_hidden_layers)


        self.dropout = nn.Dropout(0.35)
        # self.batchnorm = nn.BatchNorm1d()

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        else:
            self.activation = None

        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor):
        batch_size = X.size()[0]
        node_size = X.size()[1]
        output: List[torch.Tensor] = []

        for batch in range(batch_size):
            # calculate each graph
            train_X = X[batch]
            train_A = A[batch]

            # encoding input graph
            for layer in self.encoder_layer:
                train_X = self.dropout(self.activation(layer(train_X)))
                # train_X = self.activation(layer(train_X))

            # N iteration of processing core function
            for step in range(self.message_passing_steps):
                train_X_node = train_X.clone()
                train_X_edge = train_X.clone()

                for layer in self.node_update_layer[step]:
                    train_X_node = self.dropout(self.activation(layer(train_X_node)))
                    # train_X_node = self.activation(layer(train_X_node))

                for layer in self.edge_update_layer[step]:
                    train_X_edge = self.dropout(self.activation(layer(train_X_edge)))
                    # train_X_edge = self.activation(layer(train_X_edge))

                train_X_edge = torch.bmm(train_A.unsqueeze(0),train_X_edge.unsqueeze(0))[0]
                next_train_X = train_X_node + train_X_edge
                next_train_X /= node_size # Fully connected graph
                train_X = next_train_X

                for layer in self.node_update_layers2[step]:
                    train_X = self.dropout(self.activation(layer(train_X)))

            for layer in self.output_update_layer[:-1]:
                train_X = self.activation(layer(train_X))
            output.append(self.sigmoid(self.output_update_layer[-1](train_X)))

        return torch.stack(output)

# test
if __name__=='__main__':
    g = Graph()
    for i in range(4):
        x: list = list()
        x.append([3+i,2+i,4+i,1+i])
        x.append([2+i,4+i,1+i,5+i])
        x.append([6+i,1+i,3+i,2+i])
        y = [0,1,1]
        edge = [[0,1,1],[1,0,1],[1,1,0]]
        g.add_graph(x, edge, y)
    dataloader = data.DataLoader(g, batch_size=2, shuffle=True)
    model = GNN_clf(4, 3, 2, activation='elu')
    model.apply(init_weights)
    epochs = 300
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        for batch_idx, (X,A,y) in enumerate(dataloader):
            optimizer.zero_grad()
            prediction_y = model(X,A)
            print(prediction_y)
            loss = loss_function(prediction_y,y.unsqueeze(2))
            loss.backward()
            optimizer.step()
            print(loss.item())