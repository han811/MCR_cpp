import argparse
import logging
import os
import pickle
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

from myGNN_preprocessing import train_validation_test_data_load, indicator_coordinates_graph, all_indicator_coordinates_graph, plot_obstacle_graph_result
from myGNNs import myGCN, myGraphSAGE, myGAT, myGraphSAGE_ED
from myGNN_config import GCN_config, SAGE_config, GAT_config
from myGNN_utils import FocalLoss, get_n_params, TqdmLoggingHandler, weighted_binary_cross_entropy, count_label_01_dataset

from myModels import mySimpleFC, myAutoEncoder
from myModel_config import AE_config



'''
    This is plotting train result code
'''

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--model', required=True, default='GCN', type=str, help='choose model among GCN, SAGE, GAT, GATcVAE')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')

parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8.0, type=float, help='height size of map')

parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--all', required=False, default=False, type=bool, help='plot all error cases')


args = parser.parse_args()

model = args.model
device = args.device

width = args.width
height = args.height

delta = args.delta
ALL = args.all

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()


#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_indexs, validation_indexs, test_indexs = train_validation_test_data_load()

in_node = len(next(indicator_coordinates_graph(train_indexs, delta=delta))[0])
in_node += 4
# my_ae_model = myAutoEncoder(in_channels=in_node, **AE_config)
# path = f'./save_model/AE/feature_ae_sigmoid_mse_256_fixed_2layers_1024_huge_dataset.pt'
# my_ae_model.load_state_dict(torch.load(path))
# my_ae_model.eval()
# in_node = AE_config['hidden_channels']+4

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
is_save_hidden = False
if model=='GCN':
    mymodel = myGCN(in_channels=in_node, is_save_hiddens=is_save_hidden, **GCN_config)
elif model=='SAGE':
    mymodel = myGraphSAGE(in_channels=in_node, is_save_hiddens=is_save_hidden, **SAGE_config)
    # mymodel = myGraphSAGE_ED(in_channels=in_node, embedding_channels=64, is_save_hiddens=is_save_hidden, **SAGE_config)
elif model=='GAT':
    mymodel = myGAT(in_channels=in_node, is_save_hiddens=is_save_hidden, **GAT_config)
elif model=='FC':
    mymodel = mySimpleFC(in_channels=in_node, hidden_channels=64)
print(mymodel)
print()
print(f'number of clf model parameters: {get_n_params(mymodel)}')
print()


#########################################################################################################################################
'''  set training parameters  '''
#########################################################################################################################################
mymodel = mymodel.to(device)

n_train_set = len(train_indexs)
n_validation_set = len(validation_indexs)
n_test_set = len(test_indexs)

path = './save_model/SAGE/my_model_SAGE_one_hot_16_0.2_non_optimal.pt'
mymodel.load_state_dict(torch.load(path))
mymodel.eval()

error_cases = []
error_cases_6_6 = []
error_cases_label_6 = []
error_cases_prediction_6 = []
error_cases_prediction_6_label = []
error_cases_others = []
case_idx = 0

for input_graph in tqdm(indicator_coordinates_graph(test_indexs, width=width, height=height, delta=delta)):
    # input_graph2 = my_ae_model.enc(torch.tensor(input_graph,dtype=torch.float32)).tolist()
    input_graph2 = input_graph
    
    graph_start_goal_path = os.getcwd()
    graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{test_indexs[case_idx]}_{delta}.npy'
    start_goal = np.load(graph_start_goal_path)[0:4].tolist()

    for sub_graph in input_graph2:
        sub_graph += start_goal

    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{test_indexs[case_idx]}_{delta}.npy').tolist()
    n_obstacle = len(input_graph2)
    tmp_edge = []
    tmp_edge_element1 = []
    tmp_edge_element2 = []
    for i in range(n_obstacle):
        for j in range(n_obstacle):
            if i!=j:
                tmp_edge_element1 += [j]
        tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
    tmp_edge = [tmp_edge_element1, tmp_edge_element2]

    tmp_data = Data(x=torch.tensor(input_graph2,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32))
    tmp_data.to(device)
    prediction_y = mymodel(tmp_data.x,tmp_data.edge_index)
    prediction_y_int = prediction_y.view(-1)>0.5
    if (prediction_y_int != tmp_data.y).sum().item() > 0:
        error_cases.append(test_indexs[case_idx])
        s1 = prediction_y_int.sum().item()
        s2 = tmp_data.y.sum().item()
        if s1==6 and s2==6:
            error_cases_6_6.append(test_indexs[case_idx])
        elif s1==6:
            error_cases_prediction_6.append(test_indexs[case_idx])
            error_cases_prediction_6_label.append(tmp_data.y.tolist())
        elif s2==6:
            error_cases_label_6.append(test_indexs[case_idx])
        else:
            error_cases_others.append(test_indexs[case_idx])

    case_idx += 1
print(f'error_cases:',len(error_cases))
print(f'error_cases_6_6:',len(error_cases_6_6))
print(f'error_cases_label_6:',len(error_cases_label_6))
print(f'error_cases_prediction_6:',len(error_cases_prediction_6))
print(f'error_cases_others:',len(error_cases_others))
# print(error_cases_prediction_6_label)
sample=error_cases_others[:5]
print(sample)
if ALL:
    case_idx = 0
    for input_graph in tqdm(indicator_coordinates_graph(sample, width=width, height=height, delta=delta)):
        # input_graph2 = my_ae_model.enc(torch.tensor(input_graph,dtype=torch.float32)).tolist()
        input_graph2 = input_graph
        
        graph_start_goal_path = os.getcwd()
        graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{sample[case_idx]}_{delta}.npy'
        start_goal = np.load(graph_start_goal_path)[0:4].tolist()

        for sub_graph in input_graph2:
            sub_graph += start_goal

        tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{sample[case_idx]}_{delta}.npy').tolist()
        n_obstacle = len(input_graph2)
        tmp_edge = []
        tmp_edge_element1 = []
        tmp_edge_element2 = []
        for i in range(n_obstacle):
            for j in range(n_obstacle):
                if i!=j:
                    tmp_edge_element1 += [j]
            tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
        tmp_edge = [tmp_edge_element1, tmp_edge_element2]

        tmp_data = Data(x=torch.tensor(input_graph2,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32))
        tmp_data.to(device)
        prediction_y = mymodel(tmp_data.x,tmp_data.edge_index)
        prediction_y_int = prediction_y.view(-1)>0.5

        plot_obstacle_graph_result(sample[case_idx], prediction_y_int.tolist(), width=12.0, height=12.0, delta=0.2)
        case_idx += 1