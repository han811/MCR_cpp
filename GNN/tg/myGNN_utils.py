import os
import pickle
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Linear, BatchNorm, GCN, GraphSAGE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        # ce_loss = nn.BCELoss(reduction='none')(inputs, targets.long())
        ce_loss = nn.BCELoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * ce_loss
        return F_loss.mean()

def weighted_binary_cross_entropy(output, target, weight=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    if weight is not None:
        assert len(weight) == 2
        loss = weight[1] * (target * torch.log(output)) + weight[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = (target * torch.log(output)) + ((1 - target) * torch.log(1 - output))
    return torch.neg(torch.mean(loss))

def count_label_01_dataset(dataset):
    num_0 = 0
    num_1 = 0
    for data in dataset:
        for tmp_y in data.y:
            if tmp_y:
                num_1+=1
            else:
                num_0+=1
    print(f"count 0 : {num_0}")            
    print(f"count 1 : {num_1}")     
    print(f"ratio 0 & 1: {num_0/(num_0+num_1)} & {num_1/(num_0+num_1)}")
    return [num_0, num_1]

def init_weights(m):
    if type(m) == nn.Linear or type(m) == Linear or type(m) == GATConv or type(m) == GraphSAGE:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

    

if __name__=='__main__':
    FocalLoss()