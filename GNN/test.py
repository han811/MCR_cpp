import os, sys
from datetime import datetime
import pickle

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from model import GNN
from utils import graph_generate_load, init_weights, key_configuration_load
from config import model_test_config

# setting parameters
epochs = model_test_config['epochs']
learning_rate = model_test_config['learning_rate']
CUDA = model_test_config['CUDA']
log_step = model_test_config['log_step']
save_step = model_test_config['save_step']
TRAIN = model_test_config['TRAIN']
plot_mAP = model_test_config['plot_mAP']
probability_threshold = model_test_config['probability_threshold']
model_path = model_test_config['model_path']
PREPARE = model_test_config['PREPARE']


def plot_all_data_set(GNN_model, data, key_configurations, probability_threshold=0.5,width=12.0, height=8.0, name='validation'):
    
    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    for num in range(len(data)):

        print(f'plot image percentage: {num/len(data)}', end='\r')
        graph_, edge_, label_ = data[num]
        graph_ = graph_.unsqueeze(0)
        edge_ = edge_.unsqueeze(0)
        label_ = label_.unsqueeze(0)

        pred = GNN_model(graph_.cuda(), edge_.cuda())
        pred_label = (pred>probability_threshold).float().cuda()
        
        squeezed_label = label_.squeeze()
        squeezed_pred = pred_label.squeeze()

        for ob_idx,ob in enumerate(graph_.squeeze()):
            if squeezed_label[ob_idx]==1 and squeezed_pred[ob_idx]==1:
                for key_idx, indicator in enumerate(ob[:-4]):
                    if indicator:
                        key_configuration = key_configurations[key_idx]
                        plt.gca().scatter(key_configuration[0],key_configuration[1],c='orange',s=0.25,alpha=1.0)
            elif squeezed_label[ob_idx]==1:
                for key_idx, indicator in enumerate(ob[:-4]):
                    if indicator:
                        key_configuration = key_configurations[key_idx]
                        plt.gca().scatter(key_configuration[0],key_configuration[1],c='red',s=0.25,alpha=1.0)
            elif squeezed_pred[ob_idx]==1:
                for key_idx, indicator in enumerate(ob[:-4]):
                    if indicator:
                        key_configuration = key_configurations[key_idx]
                        plt.gca().scatter(key_configuration[0],key_configuration[1],c='blue',s=0.25,alpha=1.0)
            else:
                for key_idx, indicator in enumerate(ob[:-4]):
                    if indicator:
                        key_configuration = key_configurations[key_idx]
                        plt.gca().scatter(key_configuration[0],key_configuration[1],c='green',s=0.25,alpha=1.0)
            start_point = ob[-4:-2]
            plt.gca().scatter(start_point[0],start_point[1],c='black',s=0.45,alpha=1.0)
            goal_point = ob[-2:]
            plt.gca().scatter(goal_point[0],goal_point[1],c='black',s=0.45,alpha=1.0)
        
        plt.savefig(f'./images/{name}_{num}.png')
        plt.cla()

if __name__=='__main__':
    
    print("Data preprocessing start!!")
    _, in_node, _ = graph_generate_load()
    # graph_inputs, in_node, labels = graph_generate_load()
    in_node += 4

    GNN_model = GNN(in_node,100,in_node,3,4,cuda=CUDA)
    GNN_model.load_state_dict(torch.load(model_path))
    GNN_model.eval() 
    print("------------------------")
    print("Data preprocessing end!!")
    print()
    

    if CUDA:
        GNN_model.cuda()
    
    train_set_path = os.getcwd()
    train_set_path += '/data/train_set.pickle'
    with open(train_set_path,'rb') as f:
        train_set, train_label = pickle.load(f)

    val_set_path = os.getcwd()
    val_set_path += '/data/validation_set.pickle'
    with open(val_set_path,'rb') as f:
        val_set, val_label = pickle.load(f)

    test_set_path = os.getcwd()
    test_set_path += '/data/test_set.pickle'
    with open(test_set_path,'rb') as f:
        test_set, test_label = pickle.load(f)
    

    key_configurations = key_configuration_load()
    # plot_all_data_set(GNN_model, train_set, key_configurations, name='train')
    plot_all_data_set(GNN_model, val_set, key_configurations)
    exit()

    torch.set_printoptions(precision=3)
    # train
    with torch.no_grad():
        while_sig = True
        while while_sig:
            data_set = input(f'put which data set among train, validation and test you want to check\nif you want to quit then put None\ndata name: ')
            
            if data_set=='train':
                data = train_set
            elif data_set=='validation':
                data = val_set
            elif data_set=='test':
                data = test_set
            else:
                while_sig = False
                continue

            num = int(input(f"# of {len(data)} data: "))

            graph_, edge_, label_ = data[num]
            graph_ = graph_.unsqueeze(0)
            edge_ = edge_.unsqueeze(0)
            label_ = label_.unsqueeze(0)

            print(graph_.size())
            print(edge_.size())

            pred = GNN_model(graph_.cuda(), edge_.cuda())
            
            print_label = label_.squeeze()
            print_pred = pred.cpu().squeeze()

            n_digits = 3
            rounded_print_label = torch.round(print_label * (10**n_digits)) / (10**n_digits)
            rounded_print_pred = torch.round(print_pred * (10**n_digits)) / (10**n_digits)
            print(rounded_print_label)
            print(rounded_print_pred)
