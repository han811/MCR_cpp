import os, sys
from datetime import datetime
from typing import List, Tuple, Dict

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

from data_class import MCRdata, GraphSaveClass, MyGraphSaveClass
from GNNclf import GNN_clf

from graph import Graph
from utils import graph_generate_load, init_weights, count_label_01, key_configuration_load
from clf_config import model_config

# setting parameters
epochs = model_config['epochs']
learning_rate = model_config['learning_rate']
CUDA = model_config['CUDA']
log_step = model_config['log_step']
save_step = model_config['save_step']
TRAIN = model_config['TRAIN']
plot_mAP = model_config['plot_mAP']
probability_threshold = model_config['probability_threshold']
model_path = model_config['model_path']
batch_size = model_config['batch_size']

if __name__=='__main__':
    # make a model
    print("Data preprocessing start!!")
    graph_inputs, in_node, labels = graph_generate_load()
    in_node += 4
    print("------------------------")
    print("Data preprocessing end!!")

    # make a dataset      
    train_length = int(len(graph_inputs) * 0.7)
    train_set, val_test_set, train_label, val_test_label = train_test_split(graph_inputs, labels, test_size=0.3)
    
    val_length = int((len(graph_inputs) - train_length) * 2 / 3)
    test_length = len(graph_inputs) - train_length - val_length
    val_set, test_set, val_label, test_label = train_test_split(val_test_set, val_test_label, test_size=0.33)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    clf_model = GNN_clf(in_node, activation='elu')
    clf_model.apply(init_weights)

    optimizer = optim.Adam(clf_model.parameters(),lr=learning_rate, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    if CUDA:
        clf_model.cuda()
        loss_function.cuda()

    writer = SummaryWriter()
    ############# fix for graph #############
    # train
    if TRAIN:
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}')
            avg_loss = 0
            for batch_idx, (X,A,y) in enumerate(train_loader):
                if CUDA:
                    X = X.cuda()
                    A = A.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                prediction_y = clf_model(X,A)
                loss = loss_function(prediction_y,y.unsqueeze(2))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                print(f'each batch: {loss.item()}',end='\r')
            avg_loss /= len(train_loader)
            print(f'epoch loss: {avg_loss}')
            writer.add_scalar("epoch_loss", avg_loss, epoch)