import os
import pickle

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from torch_geometric.nn import Linear
import torch.optim as optim
from tg_Preprocessing import load_tg_data
from torch_geometric.loader import DataLoader

class myGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation='relu'):
        super(myGraphEncoder, self).__init__()

        self.Linear1 = Linear(in_channels, hidden_channels)
        self.Linear2 = Linear(hidden_channels, hidden_channels)
        self.Linear3 = Linear(hidden_channels, out_channels)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()

    def forward(self, x, edge_index):
        x = self.Linear1(x)
        x = self.activation(x)
        
        x = self.Linear2(x)
        x = self.activation(x)

        x = self.Linear3(x)

        return x

class myGraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation='relu'):
        super(myGraphDecoder, self).__init__()

        self.Linear1 = Linear(in_channels,hidden_channels)
        self.Linear2 = Linear(hidden_channels,hidden_channels)
        self.Linear3 = Linear(hidden_channels,out_channels)

        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()

    def forward(self, x, edge_index):
        x = self.Linear1(x)
        x = self.activation(x)
        
        x = self.Linear2(x)
        x = self.activation(x)

        x = self.Linear3(x)

        return x

class myGraphAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, activation='relu'):
        super(myGraphAE, self).__init__()
        self.enc = myGraphEncoder(in_channels, hidden_channels, latent_channels, activation='relu')
        self.dec = myGraphDecoder(latent_channels, hidden_channels, in_channels, activation='relu')
    
    def forward(self, x, edge_index):
        x = self.enc(x, edge_index)
        x = self.dec(x, edge_index)
        return x


if __name__=='__main__':
    train_set:list = []
    val_set:list = []
    test_set:list = []
    for data_num in range(0,3):
        tmp_train_set, tmp_val_set, tmp_test_set = load_tg_data(num=data_num)
        train_set += tmp_train_set
        val_set += tmp_val_set
        test_set += tmp_test_set

    in_node = len(train_set[0].x[0])

    index = []
    for idx,t in enumerate(train_set):
        if t.x.size()[1]!=in_node:
            index.append(idx)
    index.reverse()
    for t in index:
        del train_set[t]

    index = []
    for idx,t in enumerate(val_set):
        if t.x.size()[1]!=in_node:
            index.append(idx)
    index.reverse()
    for t in index:
        del val_set[t]

    index = []
    for idx,t in enumerate(test_set):
        if t.x.size()[1]!=in_node:
            index.append(idx)
    index.reverse()
    for t in index:
        del test_set[t]

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    model = myGraphAE(in_node,2048,512,'elu')




    epochs = 100
    device = 'cuda'
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    loss_function = nn.MSELoss()

    model = model.to(device)
    loss_function = loss_function.to(device)

    n_train_set = len(train_set)

    for epoch in range(epochs):
        print(f'epoch: {epoch+1}')
        avg_mse = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction_x = model(data.x,data.edge_index)
            loss = loss_function(reconstruction_x,data.x)
            loss.backward()
            optimizer.step()
            avg_mse += loss.item()
            
            print(f'each mse: {loss.item()}',end='\r')
        print(f'epoch mse: {avg_mse/n_train_set}')
    
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                reconstruction_x = model(data.x,data.edge_index)
                print(reconstruction_x)
                print(data.x)