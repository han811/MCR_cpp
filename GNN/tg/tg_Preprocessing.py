import os
import pickle
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def preprocess_tg_data(num=0):
    # train
    train_set_path = '/'.join(os.getcwd().split('/')[:-1])
    train_set_path += f'/data/train_set{num}.pickle'
    with open(train_set_path,'rb') as f:
        train_set, _ = pickle.load(f)

    train_total_dataset = list()

    for i in range(len(train_set)):
        x = train_set[i][0]
        y = train_set[i][2]
        edge_index1 = list()
        edge_index2 = list()
        for j in range(len(x)):
            for k in range(len(x)):
                if j!=k:
                    edge_index1.append(j) 
                    edge_index2.append(k)
        edge_index = [edge_index1, edge_index2]
        edge_index = torch.LongTensor(edge_index)
        tmp_data = Data(x=x, edge_index=edge_index, y=y)
        train_total_dataset.append(tmp_data)
    
    tg_train_set_path = '/'.join(os.getcwd().split('/')[:-1])
    tg_train_set_path += f'/data/tg_train_set{num}.pickle'

    with open(tg_train_set_path, 'wb') as f:
        pickle.dump(train_total_dataset, f, pickle.HIGHEST_PROTOCOL)

    # validation
    val_set_path = '/'.join(os.getcwd().split('/')[:-1])
    val_set_path += f'/data/validation_set{num}.pickle'
    with open(val_set_path,'rb') as f:
        val_set, _ = pickle.load(f)

    val_total_dataset = list()

    for i in range(len(val_set)):
        x = val_set[i][0]
        y = val_set[i][2]
        edge_index1 = list()
        edge_index2 = list()
        for j in range(len(x)):
            for k in range(len(x)):
                if j!=k:
                    edge_index1.append(j) 
                    edge_index2.append(k)
        edge_index = [edge_index1, edge_index2]
        edge_index = torch.LongTensor(edge_index)
        tmp_data = Data(x=x, edge_index=edge_index, y=y)
        val_total_dataset.append(tmp_data)
    
    tg_val_set_path = '/'.join(os.getcwd().split('/')[:-1])
    tg_val_set_path += f'/data/tg_validation_set{num}.pickle'

    with open(tg_val_set_path, 'wb') as f:
        pickle.dump(val_total_dataset, f, pickle.HIGHEST_PROTOCOL)

    # test
    test_set_path = '/'.join(os.getcwd().split('/')[:-1])
    test_set_path += f'/data/test_set{num}.pickle'
    with open(test_set_path,'rb') as f:
        test_set, _ = pickle.load(f)

    test_total_dataset = list()

    for i in range(len(test_set)):
        x = test_set[i][0]
        y = test_set[i][2]
        edge_index1 = list()
        edge_index2 = list()
        for j in range(len(x)):
            for k in range(len(x)):
                if j!=k:
                    edge_index1.append(j) 
                    edge_index2.append(k)
        edge_index = [edge_index1, edge_index2]
        edge_index = torch.LongTensor(edge_index)
        tmp_data = Data(x=x, edge_index=edge_index, y=y)
        test_total_dataset.append(tmp_data)
    
    tg_test_set_path = '/'.join(os.getcwd().split('/')[:-1])
    tg_test_set_path += f'/data/tg_test_set{num}.pickle'

    with open(tg_test_set_path, 'wb') as f:
        pickle.dump(test_total_dataset, f, pickle.HIGHEST_PROTOCOL)

def load_tg_data(num=0):

    train_set_path = '/'.join(os.getcwd().split('/')[:-1])
    train_set_path += f'/data/tg_train_set{num}.pickle'
    with open(train_set_path,'rb') as f:
        train_set = pickle.load(f)

    val_set_path = '/'.join(os.getcwd().split('/')[:-1])
    val_set_path += f'/data/tg_validation_set{num}.pickle'
    with open(val_set_path,'rb') as f:
        val_set = pickle.load(f)

    test_set_path = '/'.join(os.getcwd().split('/')[:-1])
    test_set_path += f'/data/tg_test_set{num}.pickle'
    with open(test_set_path,'rb') as f:
        test_set = pickle.load(f)
    
    return train_set, val_set, test_set

if __name__=='__main__':
    for i in range(6):
        preprocess_tg_data(num=i)
        tg_train_set, tg_validation_set, tg_test_set = load_tg_data(num=i)
        print(i)