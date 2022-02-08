import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from GNN import Node_Update_Function, Edge_to_Node_Aggregation_Function

DEVICE = "cpu" # "cuda"
device = torch.device(DEVICE)

class GAN_GNN_Generator(nn.Module):
    def __init__(self, in_f, z_dim, hidden_fs):
        super(GAN_GNN_Generator, self).__init__()
        self.in_f = in_f
        self.z_dim = z_dim
        self.hidden_fs = hidden_fs
        
    def _build(self):
        self.model = [nn.Linear(self.in_f+self.z_dim, self.hidden_fs[i]) if i==0 else
            nn.Linear(self.hidden_fs[-1],1) if i==len(self.hidden_fs) else
            nn.Linear(self.hidden_fs[i-1],self.hidden_fs[i]) for i in range(len(self.hidden_fs)+1)]
        self.relu = nn.ReLU()
    
    def forward(self, x, z):
        for i in range(len(self.hidden_fs)):
            x = self.relu(self.model[i](x))
        return self.model[-1](x)

class GAN_GNN_Discriminator(nn.Module):
    def __init__(self, node_num, in_f, hidden_fs):
        super(GAN_GNN_Discriminator, self).__init__()
        self.node_num = node_num
        self.in_f = in_f
        self.hidden_fs = hidden_fs

    def _build(self):
        self.model = [nn.Linear(self.in_f * 2, self.hidden_fs[i]) if i==0 else
            nn.Linear(self.hidden_fs[-1],1) if i==len(self.hidden_fs) else
            nn.Linear(self.hidden_fs[i-1],self.hidden_fs[i]) for i in range(len(self.hidden_fs)+1)]
        self.relu = nn.ReLU()
        self.clf = nn.Sigmoid()

        self.Node_update_layers = [Node_Update_Function(self.in_f,self.in_f,(64,128,64)),
            Node_Update_Function(self.in_f,self.in_f,(64,128,64))]
        self.Edge_update_layers = [Edge_to_Node_Aggregation_Function(self.in_f, 64, self.in_f,(64,32),(64,32)),
            Edge_to_Node_Aggregation_Function(self.in_f, 64, self.in_f,(64,32),(64,32))]

    def forward(self, x, l, adjancy_matrix):
        h_x = x 
        for i in range(2):
            h_x = self.Edge_update_layers[i](h_x, adjancy_matrix)
            h_x = torch.cat([x, h_x], dim=2)
            h_x = self.Node_update_layers[i](h_x)
        x = torch.sum(x,0)
        x /= self.node_num
        x = torch.cat([x, l], dim=2)
        for i in range(len(self.hidden_fs)):
            x = self.relu(self.model[i](x))
        return self.clf(self.model[-1](x))




if __name__=='__main__':
    GAN_GNN_Generator(3,4,[64,128,64])
    GAN_GNN_Discriminator(16,32,[64,128,64])
