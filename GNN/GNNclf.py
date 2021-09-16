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
class GNN_clf(nn.Module):
    def __init__(self, n_node_feature,
     node_hidden_layers = (128, 256, 128),
     edge_hidden_layers = (128, 256, 128),
     output_hidden_layers = (128, 256, 128),
     message_passing_steps=3, activation=None):
        super(GNN_clf, self).__init__()

        self.n_node_feature = n_node_feature

        self.node_hidden_layers = node_hidden_layers
        self.edge_hidden_layers = edge_hidden_layers
        self.output_hidden_layers = output_hidden_layers

        self.node_update_layer, _ = layer_generator(self.n_node_feature,self.n_node_feature,self.node_hidden_layers)
        self.edge_update_layer, _ = layer_generator(self.n_node_feature,self.n_node_feature,self.edge_hidden_layers)
        self.output_update_layer, _ = layer_generator(self.n_node_feature,1,self.output_hidden_layers)

        self.message_passing_steps = message_passing_steps

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        else:
            self.activation = None

        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor):
        ''' Assume a Fully-Connected Graph '''
        batch_size = X.size()[0]
        output: List[torch.Tensor] = []
        for batch in range(batch_size):
            train_X = X[batch]
            train_A = A[batch]
            for step in range(self.message_passing_steps):
                train_X_node = train_X.clone().detach()
                train_X_edge = train_X.clone().detach()
                for layer in self.node_update_layer:
                    train_X_node = self.activation(layer(train_X_node))
                for layer in self.edge_update_layer:
                    train_X_edge = self.activation(layer(train_X_edge))
                next_train_X: List[torch.Tensor] = []
                for node_idx in range(train_X_node.size()[0]):
                    tmp_X_node = train_X_node[node_idx]
                    for edge_idx in range(train_X_node.size()[0]):
                        if edge_idx!=node_idx:
                            tmp_X_node += train_X_edge[edge_idx]
                    tmp_X_node /= train_X_node.size()[0]
                    next_train_X.append(tmp_X_node)
                train_X = torch.stack(next_train_X)
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
    model = GNN_clf(4, activation='elu')
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
            loss = loss_function(prediction_y,y.unsqueeze(2))
            loss.backward()
            optimizer.step()
            print(loss.item())