import argparse
import logging
import os
import pickle
import random
from sys import path
from time import sleep
from datetime import datetime
from typing import *

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data.data import Data
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from myGNN_preprocessing import train_validation_test_data, indicator_coordinates_graph, all_indicator_coordinates_graph
from myGNNs import myGCN, myGraphSAGE, myGAT, myGATcVAE, mySAGEcVAE, myGraphSAGE_ED
from myGNN_config import GCN_config, SAGE_config, GAT_config, GATcVAE_config,GAE_config
from myGNN_utils import FocalLoss, get_n_params, TqdmLoggingHandler, weighted_binary_cross_entropy, count_label_01_dataset




'''
    This is train code for one-hot key configuration and indicator-configuration concatenated feature vector
'''

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--model', required=True, default='GCN', type=str, help='choose model among GCN, SAGE, GAT, GATcVAE')
parser.add_argument('--epochs', required=False, default=512, type=int, help='num of epochs')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')
parser.add_argument('--log_step', required=False, default=100, type=int, help='show the training log per each log step')
parser.add_argument('--save_step', required=False, default=100, type=int, help='setting model saving step')
parser.add_argument('--train', required=False, default=False, type=bool, help='whether train or evaluate')
parser.add_argument('--learning_rate', required=False, default=1e-2, type=float, help='learning rate of training')
parser.add_argument('--probabilistic_threshold', required=False, default=0.5, type=float, help='setting probabilistic threshold')
parser.add_argument('--batch_size', required=False, default=32, type=int, help='setting batch size')

parser.add_argument('--tensorboard', required=False, default=False, type=bool, help='record on tensorboard or not')
parser.add_argument('--is_save_weight', required=False, default=False, type=bool, help='record weights or not')
parser.add_argument('--is_save_hidden', required=False, default=False, type=bool, help='record hiddens or not')

parser.add_argument('--problem_type', required=True, default='clf', type=str, help='setting problem type')

parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8.0, type=float, help='height size of map')

parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--total', required=False, default=False, type=bool, help='choose total dataset or not')


args = parser.parse_args()

model = args.model
epochs = args.epochs
device = args.device
log_step = args.log_step
save_step = args.save_step
TRAIN = args.train
learning_rate = args.learning_rate
probabilistic_threshold = args.probabilistic_threshold
batch_size = args.batch_size

is_tensorboard = args.tensorboard
is_save_weight = args.is_save_weight
is_save_hidden = args.is_save_hidden

problem_type = args.problem_type

width = args.width
height = args.height

delta = args.delta
total = args.total

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()

#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_indexs, validation_indexs, test_indexs = train_validation_test_data((0.7, 0.2, 0.1))

in_node = len(next(indicator_coordinates_graph(train_indexs, delta=delta))[0])
in_node += 4

print(f'train : {len(train_indexs)}\nvalidation : {len(validation_indexs)}\ntest : {len(test_indexs)}')
print()
print(f'node feature size : {in_node}')
print()

#########################################################################################################################
'''  select which model to train  '''
#########################################################################################################################################
'''
    you have to set model layer parameters in myGNN_config.py file
'''
if model=='GCN':
    mymodel = myGCN(in_channels=in_node, is_save_hiddens=is_save_hidden, **GCN_config)
elif model=='SAGE':
    mymodel = myGraphSAGE(in_channels=in_node, is_save_hiddens=is_save_hidden, **SAGE_config)
    # mymodel = myGraphSAGE_ED(in_channels=in_node, embedding_channels=64, is_save_hiddens=is_save_hidden, **SAGE_config)
elif model=='GAT':
    mymodel = myGAT(in_channels=in_node, is_save_hiddens=is_save_hidden, **GAT_config)
elif model=='SAGEcVAE':
    mymodel = mySAGEcVAE(en_in_channels=in_node, de_in_channels=in_node, is_save_hiddens=is_save_hidden, **GATcVAE_config)
    z_dim = GATcVAE_config['z_dim']
elif model=='GATcVAE':
    mymodel = myGATcVAE(en_in_channels=in_node, de_in_channels=in_node, is_save_hiddens=is_save_hidden, **GATcVAE_config)
    z_dim = GATcVAE_config['z_dim']
print(f'{problem_type} model')
print(mymodel)
print()
print(f'number of {problem_type} model parameters: {get_n_params(mymodel)}')
print()

#########################################################################################################################################
'''  set training parameters  '''
#########################################################################################################################################
# optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate, weight_decay=0.9)
optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate)

if model=='GCN' or model=='SAGE' or model=='GAT':
    # loss_function = nn.BCELoss()
    loss_function = nn.BCELoss(reduction='none')
    # loss_function = FocalLoss(gamma=3.5,alpha=20)
    # loss_function = weighted_binary_cross_entropy
elif model=='SAGEcVAE' or model=='GATcVAE':
    loss_function = nn.MSELoss()

mymodel = mymodel.to(device)
loss_function = loss_function.to(device)

if is_save_weight:
    with open(f'./weights/{model}/weights_initialization','wb') as f:
        pickle.dump(mymodel.state_dict(),f,pickle.HIGHEST_PROTOCOL)

if is_tensorboard:
    num = len(os.listdir(os.getcwd()+f'/GNN_model_train/{model}'))
    writer = SummaryWriter(f'GNN_model_train/{model}/{num}')

n_train_set = len(train_indexs)
n_validation_set = len(validation_indexs)
n_test_set = len(test_indexs)


#########################################################################################################################################
'''  setting for logger  '''
#########################################################################################################################################
log = logging.getLogger(__name__)
log.handlers = []
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

if total:
    train_loader = DataLoader(all_indicator_coordinates_graph(train_indexs), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(all_indicator_coordinates_graph(validation_indexs), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(all_indicator_coordinates_graph(test_indexs), batch_size=batch_size, shuffle=True)

#########################################################################################################################################
'''  start training  '''
#########################################################################################################################################
weight = [0.1, 0.9]

if TRAIN:
    # setting for save best model
    pre_accuracy = 0
    current_accuracy = 0

    for epoch in tqdm(range(epochs), desc='Epoch'):
        if problem_type=='clf':
            avg_loss = 0
            avg_accuracy = 0
            avg_0_accuracy = 0
            avg_1_accuracy = 0
            count_0 = 0
            count_1 = 0
        elif problem_type=='generation':
            avg_mse = 0
            avg_kl = 0
            avg_loss = 0

        if total:
            for batch_idx, batch_data in tqdm(enumerate(train_loader), desc='train'):
                n_obstacle = batch_data.x.size()[0]
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                if model=='GCN' or model=='SAGE':
                    prediction_y = mymodel(batch_data.x,batch_data.edge_index)
                elif model=='GAT':
                    prediction_y, attentions = mymodel(batch_data.x,batch_data.edge_index)
                elif model=='SAGEcVAE' or model=='GATcVAE':
                    reconstruction_y, z_mu, z_log_var = mymodel(batch_data.x,batch_data.edge_index,batch_data.y)

                if problem_type=='clf':
                    loss = loss_function(prediction_y,batch_data.y.unsqueeze(1))
                    loss = loss[0]*weight[0]+loss[1]*weight[1]
                    # loss = loss_function(prediction_y,batch_data.y.unsqueeze(1),weight=[0.2,0.8])
                elif problem_type=='generation':
                    mse = loss_function(reconstruction_y,batch_data.y.unsqueeze(1))
                    kl = (-0.5 * (1 + z_log_var - z_mu.pow(2) - z_log_var.exp()).sum())
                    loss = mse + kl
                    
                loss.backward()
                optimizer.step()
                
                if problem_type=='clf':
                    avg_loss += loss.item()
                elif problem_type=='generation':
                    avg_loss += loss.item()
                    avg_mse += mse.item()
                    avg_kl += kl.item()
                    
                if problem_type=='clf':
                    prediction_y_acc = prediction_y>probabilistic_threshold
                    avg_accuracy += (prediction_y_acc == batch_data.y.unsqueeze(1)).sum().item()
                    count_0 += batch_data.y.size()[0] - batch_data.y.count_nonzero()
                    count_1 += batch_data.y.count_nonzero()
                    for index, tmp_y in enumerate(batch_data.y):
                        if tmp_y.item()==0:
                            if tmp_y==prediction_y_acc[index]:
                                avg_0_accuracy += 1
                        elif tmp_y.item()==1:
                            if tmp_y==prediction_y_acc[index]:
                                avg_1_accuracy += 1
                elif problem_type=='generation':
                    # print(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}',end='\r')
                    log.info(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}')
                
                if batch_idx % log_step == 0:
                    log.info(f'each batch loss: {loss.item():0.5f}')
        else:
            for idx in tqdm(range(0,len(train_indexs),batch_size), leave=False, desc='Batch'):
                current_train_indexs = train_indexs[idx*batch_size:(idx+1)*batch_size]
                current_data = []
                for idx2, input_graph in enumerate(indicator_coordinates_graph(current_train_indexs,width=width,height=height,delta=delta)):
                    graph_start_goal_path = os.getcwd()
                    graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{train_indexs[idx2]}_{delta}.npy'
                    start_goal = np.load(graph_start_goal_path)[0:4].tolist()

                    for sub_graph in input_graph:
                        sub_graph += start_goal
                    
                    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{train_indexs[idx2]}_{delta}.npy').tolist()
                    n_obstacle = len(input_graph)
                    tmp_edge = []
                    tmp_edge_element1 = []
                    tmp_edge_element2 = []
                    for i in range(n_obstacle):
                        for j in range(n_obstacle):
                            if i!=j:
                                tmp_edge_element1 += [j]
                        tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
                    tmp_edge = [tmp_edge_element1, tmp_edge_element2]
                    current_data.append(Data(x=torch.tensor(input_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32)))
                if len(current_data)==batch_size:
                    current_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True)
                elif len(current_data)!=batch_size and len(current_data)>0:
                    current_loader = DataLoader(current_data, batch_size=len(current_data), shuffle=True)

                for batch_idx, batch_data in enumerate(current_loader):
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    if model=='GCN' or model=='SAGE':
                        prediction_y = mymodel(batch_data.x,batch_data.edge_index)
                    elif model=='GAT':
                        prediction_y, attentions = mymodel(batch_data.x,batch_data.edge_index)
                    elif model=='SAGEcVAE' or model=='GATcVAE':
                        reconstruction_y, z_mu, z_log_var = mymodel(batch_data.x,batch_data.edge_index,batch_data.y)

                    if problem_type=='clf':
                        loss = loss_function(prediction_y,batch_data.y.unsqueeze(1))
                        loss = loss[0]*weight[0]+loss[1]*weight[1]

                        # loss = loss[0]*0.2+loss[1]*0.8
                        # loss = loss_function(prediction_y,batch_data.y.unsqueeze(1),weight=[0.2,0.8])

                    elif problem_type=='generation':
                        mse = loss_function(reconstruction_y,batch_data.y.unsqueeze(1))
                        kl = (-0.5 * (1 + z_log_var - z_mu.pow(2) - z_log_var.exp()).sum())
                        loss = mse + kl
                    
                    loss.backward()
                    optimizer.step()
                    
                    if problem_type=='clf':
                        avg_loss += loss.item()
                    elif problem_type=='generation':
                        avg_loss += loss.item()
                        avg_mse += mse.item()
                        avg_kl += kl.item()
                    
                    if problem_type=='clf':
                        prediction_y_acc = prediction_y>probabilistic_threshold
                        avg_accuracy += (prediction_y_acc == batch_data.y.unsqueeze(1)).sum().item()
                        count_0 += batch_data.y.size()[0] - batch_data.y.count_nonzero()
                        count_1 += batch_data.y.count_nonzero()
                        for index, tmp_y in enumerate(batch_data.y):
                            if tmp_y.item()==0:
                                if tmp_y==prediction_y_acc[index]:
                                    avg_0_accuracy += 1
                            elif tmp_y.item()==1:
                                if tmp_y==prediction_y_acc[index]:
                                    avg_1_accuracy += 1
                    elif problem_type=='generation':
                        # print(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}',end='\r')
                        log.info(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}')
                    
                    if batch_idx % log_step == 0:
                        log.info(f'each batch loss: {loss.item():0.5f}')
        
        if problem_type=='clf':
            avg_loss /= len(range(0,len(train_indexs),batch_size))
            avg_accuracy /= (n_train_set*n_obstacle)
            avg_0_accuracy /= count_0
            avg_1_accuracy /= count_1
            print(f'epoch loss: {avg_loss}')
            print(f'epoch accuracy: {avg_accuracy*100}%')
            print(f'epoch 0 accuracy (specificity): {avg_0_accuracy*100}%')
            print(f'epoch 1 accuracy (recall): {avg_1_accuracy*100}%')
            if is_tensorboard:
                writer.add_scalar("my_GNN_epoch_loss", avg_loss, epoch)
                writer.add_scalar("my_GNN_epoch_accuracy", avg_accuracy, epoch)
                writer.add_scalar("my_GNN_epoch_0_accuracy", avg_0_accuracy, epoch)
                writer.add_scalar("my_GNN_epoch_1_accuracy", avg_1_accuracy, epoch)
        elif problem_type=='generation':
            avg_mse /= len(range(0,len(train_indexs),batch_size))
            avg_kl /= len(range(0,len(train_indexs),batch_size))
            avg_loss /= len(range(0,len(train_indexs),batch_size))
            if is_tensorboard:
                writer.add_scalar("my_GNN_epoch_loss", avg_loss, epoch)
                writer.add_scalar("my_GNN_epoch_mse", avg_mse, epoch)
                writer.add_scalar("my_GNN_epoch_kl", avg_kl, epoch)
        if is_save_weight:
            for tag, value in mymodel.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)
        
        # test part
        if problem_type=='clf':
            test_avg_loss = 0
            test_avg_accuracy = 0
            test_avg_0_accuracy = 0
            test_avg_1_accuracy = 0
            test_count_0 = 0
            test_count_1 = 0
        elif problem_type=='generation':
            test_avg_mse = 0
            test_avg_kl = 0
            test_avg_loss = 0

        with torch.no_grad():
            random_sample = random.randint(0,len(test_indexs))
            y_probs = []
            y_labels = []

            if total:
                for batch_idx, batch_data in tqdm(enumerate(test_loader), desc='test'):
                    n_obstacle = batch_data.x.size()[0]
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    if model=='GCN' or model=='SAGE':
                        prediction_y = mymodel(batch_data.x,batch_data.edge_index)
                    elif model=='GAT':
                        prediction_y, attentions = mymodel(batch_data.x,batch_data.edge_index)
                    elif model=='SAGEcVAE' or model=='GATcVAE':
                        reconstruction_y, z_mu, z_log_var = mymodel(batch_data.x,batch_data.edge_index,batch_data.y)

                    if problem_type=='clf':
                        loss = loss_function(prediction_y,batch_data.y.unsqueeze(1))
                        loss = loss[0]*weight[0]+loss[1]*weight[1]

                        # loss = loss[0]*0.2+loss[1]*0.8
                        # loss = loss_function(prediction_y,batch_data.y.unsqueeze(1),weight=[0.2,0.8])

                    elif problem_type=='generation':
                        mse = loss_function(reconstruction_y,batch_data.y.unsqueeze(1))
                        kl = (-0.5 * (1 + z_log_var - z_mu.pow(2) - z_log_var.exp()).sum())
                        loss = mse + kl
                        
                    loss.backward()
                    optimizer.step()
                    
                    if problem_type=='clf':
                        avg_loss += loss.item()
                    elif problem_type=='generation':
                        avg_loss += loss.item()
                        avg_mse += mse.item()
                        avg_kl += kl.item()
                        
                    if problem_type=='clf':
                        prediction_y_acc = prediction_y>probabilistic_threshold
                        avg_accuracy += (prediction_y_acc == batch_data.y.unsqueeze(1)).sum().item()
                        count_0 += batch_data.y.size()[0] - batch_data.y.count_nonzero()
                        count_1 += batch_data.y.count_nonzero()
                        for index, tmp_y in enumerate(batch_data.y):
                            if tmp_y.item()==0:
                                if tmp_y==prediction_y_acc[index]:
                                    avg_0_accuracy += 1
                            elif tmp_y.item()==1:
                                if tmp_y==prediction_y_acc[index]:
                                    avg_1_accuracy += 1
                    elif problem_type=='generation':
                        log.info(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}')
                    
                    if problem_type == 'clf':
                        loss = loss_function(prediction_y,batch_data.y.unsqueeze(1))
                        loss = loss[0]*weight[0]+loss[1]*weight[1]

                        # loss = loss[0]*0.2+loss[1]*0.8

                        test_avg_loss += loss.item()
                        prediction_y_acc = prediction_y>probabilistic_threshold
                        test_avg_accuracy += (prediction_y_acc == batch_data.y.unsqueeze(1)).sum().item()
                        test_count_0 += batch_data.y.size()[0] - batch_data.y.count_nonzero()
                        test_count_1 += batch_data.y.count_nonzero()
                        for index, tmp_y in enumerate(batch_data.y):
                            if tmp_y.item()==0:
                                if tmp_y==prediction_y_acc[index]:
                                    test_avg_0_accuracy += 1
                            elif tmp_y.item()==1:
                                if tmp_y==prediction_y_acc[index]:
                                    test_avg_1_accuracy += 1
                        y_probs += prediction_y.cpu().view(-1).tolist()
                        y_labels += batch_data.y.cpu().tolist()
                        log.info(f'test each batch loss: {loss.item():0.5f}')
            else:
                for idx in tqdm(range(0,len(test_indexs),batch_size), leave=False, desc='Test Batch'):

                    current_test_indexs = test_indexs[idx*batch_size:(idx+1)*batch_size]
                    current_data = []
                    for idx2, input_graph in enumerate(indicator_coordinates_graph(current_test_indexs,width=width,height=height,delta=delta)):
                        graph_start_goal_path = os.getcwd()
                        graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{test_indexs[idx2]}_{delta}.npy'
                        start_goal = np.load(graph_start_goal_path)[0:4].tolist()
                    
                    
                        for sub_graph in input_graph:
                            sub_graph += start_goal
                        tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{test_indexs[idx]}_{delta}.npy').tolist()
                        n_obstacle = len(input_graph)
                        tmp_edge = []
                        tmp_edge_element1 = []
                        tmp_edge_element2 = []
                        for i in range(n_obstacle):
                            for j in range(n_obstacle):
                                if i!=j:
                                    tmp_edge_element1 += [j]
                            tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
                        tmp_edge = [tmp_edge_element1, tmp_edge_element2]
                        current_data.append(Data(x=torch.tensor(input_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32)))
                    
                    if len(current_data)==batch_size:
                        current_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True)
                    elif len(current_data)!=batch_size and len(current_data)>0:
                        current_loader = DataLoader(current_data, batch_size=len(current_data), shuffle=True)

                    for batch_data in current_loader:
                        batch_data = batch_data.to(device)
                        if model=='GCN' or model=='SAGE':
                            prediction_y = mymodel(batch_data.x,batch_data.edge_index)
                        elif model=='GAT':
                            prediction_y, attentions = mymodel(batch_data.x,batch_data.edge_index)
                        elif model=='SAGEcVAE' or model=='GATcVAE':
                            reconstruction_y, z_mu, z_log_var = mymodel(batch_data.x,batch_data.edge_index,batch_data.y)
                        
                        if problem_type == 'clf':
                            loss = loss_function(prediction_y,batch_data.y.unsqueeze(1))
                            loss = loss[0]*weight[0]+loss[1]*weight[1]

                            # loss = loss[0]*0.2+loss[1]*0.8

                            # loss = loss_function(prediction_y,batch_data.y.unsqueeze(1),weight=[0.2,0.8])

                            test_avg_loss += loss.item()
                            prediction_y_acc = prediction_y>probabilistic_threshold
                            test_avg_accuracy += (prediction_y_acc == batch_data.y.unsqueeze(1)).sum().item()
                            test_count_0 += batch_data.y.size()[0] - batch_data.y.count_nonzero()
                            test_count_1 += batch_data.y.count_nonzero()
                            for index, tmp_y in enumerate(batch_data.y):
                                if tmp_y.item()==0:
                                    if tmp_y==prediction_y_acc[index]:
                                        test_avg_0_accuracy += 1
                                elif tmp_y.item()==1:
                                    if tmp_y==prediction_y_acc[index]:
                                        test_avg_1_accuracy += 1
                            y_probs += prediction_y.cpu().view(-1).tolist()
                            y_labels += batch_data.y.cpu().tolist()
                            log.info(f'test each batch loss: {loss.item():0.5f}')
                        
            
            if problem_type == 'clf':
                test_avg_loss /= len(range(0,len(test_indexs),batch_size))
                test_avg_accuracy /= (n_test_set*n_obstacle)
                test_avg_0_accuracy /= test_count_0
                print(f'test_avg_1_accuracy: {test_avg_1_accuracy}')
                print(f'test_count_1: {test_count_1}')
                test_avg_1_accuracy /= test_count_1

                print(f'test epoch loss: {test_avg_loss}')
                print(f'test epoch accuracy: {test_avg_accuracy*100}%')
                print(f'test epoch 0 accuracy (specificity): {test_avg_0_accuracy*100}%')
                print(f'test epoch 1 accuracy (recall): {test_avg_1_accuracy*100}%')
                

                y_labels = list(map(int,y_labels))
                
                fpr, tpr, thresholds = roc_curve(y_labels, y_probs, pos_label=1)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.savefig(f'roc/{model}/{epoch}_{get_n_params(mymodel)}_{delta}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.png')
                
                precisions, recalls, thresholds = precision_recall_curve(y_labels,y_probs,pos_label=1)
                plt.figure()
                plt.plot(recalls, precisions, color='darkorange', label='PR curve (area = %0.2f)' % average_precision_score(y_labels,y_probs))
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('PR curve')
                plt.legend(loc="lower right")
                plt.savefig(f'PR/{model}/{epoch}_{get_n_params(mymodel)}_{delta}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.png')
                

                if is_tensorboard:
                    writer.add_scalar("test_epoch_loss", test_avg_loss, epoch)
                    writer.add_scalar("test_epoch_accuracy", test_avg_accuracy, epoch)
                    writer.add_scalar("test_epoch_0_accuracy", test_avg_0_accuracy, epoch)
                    writer.add_scalar("test_epoch_1_accuracy", test_avg_1_accuracy, epoch)
        if is_tensorboard:
            for name, param in mymodel.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch)
        if problem_type=='clf':
            if epoch==0:
                current_accuracy = test_avg_accuracy
            else:
                pre_accuracy = current_accuracy
                current_accuracy = test_avg_accuracy
        
            if current_accuracy > pre_accuracy:
                torch.save(mymodel.state_dict(), f'./save_model/{model}/my_model_{model}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
        '''
            don't forget rename the model after training!!!!!
        '''
        if is_save_weight:
            with open(f'./weights/{model}/weights_{epoch}','wb') as f:
                pickle.dump(mymodel.state_dict(),f,pickle.HIGHEST_PROTOCOL)
    torch.save(mymodel.state_dict(), f'./save_model/{model}/final_my_model_{model}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')

else:
    path = './save_model/SAGE/my_model_SAGE_2021-10-11_15-27.pt'
    mymodel.load_state_dict(torch.load(path))
    mymodel.eval()
    idx = int(input(f'{len(test_indexs)}: '))
    input_graph = next(indicator_coordinates_graph(test_indexs,width=width,height=height,delta=delta))

    graph_start_goal_path = os.getcwd()
    graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{test_indexs[idx]}_{delta}.npy'
    start_goal = np.load(graph_start_goal_path)[0:4].tolist()

    for sub_graph in input_graph:
        sub_graph += start_goal
                
    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{test_indexs[idx]}_{delta}.npy').tolist()
    n_obstacle = len(input_graph)
    tmp_edge = []
    tmp_edge_element1 = []
    tmp_edge_element2 = []
    for i in range(n_obstacle):
        for j in range(n_obstacle):
            if i!=j:
                tmp_edge_element1 += [j]
        tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
    tmp_edge = [tmp_edge_element1, tmp_edge_element2]

    tmp_data = Data(x=torch.tensor(input_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32))
    tmp_data.to(device)
    prediction_y = mymodel(tmp_data.x,tmp_data.edge_index)
    print('prediction:',prediction_y.view(-1).tolist())
    print('true:',tmp_y)