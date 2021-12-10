import argparse
import logging
import os
import pickle
from sys import path
from socket import *
from datetime import datetime
from typing import *

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data.data import Data
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader

from myGNN_preprocessing import train_validation_test_data_load, indicator_coordinates_graph, all_indicator_coordinates_graph, plot_obstacle_graph_result
from myGNNs import mySAGEcVAE
from myGNN_config import SAGEcVAE_config
from myGNN_utils import get_n_params, TqdmLoggingHandler

'''
FC layer autoencoder and its configs
'''
from myModels import myAutoEncoder
from myModel_config import AE_config


# 0.2_is_optimal_False_my_model_SAGEcVAE_beta_1_z_dim_8_embedding_channels_4_en_hidden_channels_64_de_hidden_channels_4_lr_0.01_batch_128_activation_elu_parametes_13521_ntrain_19279_nvalidation_0_ntest_2143_2021-12-08_14-04.pt
'''
    This is train code for one-hot key configuration and indicator-configuration concatenated feature vector
'''

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--model', required=True, default='SAGEcVAE', type=str, help='choose model among GCN, SAGE, GAT, GATcVAE, SAGEcVAE')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')
parser.add_argument('--probabilistic_threshold', required=False, default=0.5, type=float, help='setting probabilistic threshold')
parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8.0, type=float, help='height size of map')
parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--is_optimal', required=False, default=False, type=bool, help='train model on optimal or non_optimal')

args = parser.parse_args()

model = args.model
device = args.device
probabilistic_threshold = args.probabilistic_threshold
width = args.width
height = args.height
delta = args.delta
is_optimal = args.is_optimal

beta = 1

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()

#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(('', 8080))
serverSock.listen(1)
connectionSock, addr = serverSock.accept()
print(str(addr),'address is checked')


#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_indexs, validation_indexs, test_indexs = train_validation_test_data_load(is_optimal=is_optimal)

in_node = len(next(indicator_coordinates_graph([train_indexs[0]], delta=delta, is_optimal=is_optimal))[0])
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
if model=='SAGEcVAE':
    mymodel = mySAGEcVAE(embedding_in_channels=in_node, **SAGEcVAE_config)
print(mymodel)
print()
print(f'number of clf model parameters: {get_n_params(mymodel)}')
print()

activation = SAGEcVAE_config['activation']
#########################################################################################################################################
'''  set training parameters  '''
#########################################################################################################################################
mymodel = mymodel.to(device)

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

path = './save_model/SAGEcVAE/0.2_is_optimal_False_my_model_SAGEcVAE_beta_1_z_dim_8_embedding_channels_4_en_hidden_channels_64_de_hidden_channels_32_lr_0.01_batch_128_activation_elu_parametes_14109_ntrain_3140_nvalidation_898_ntest_449_2021-11-22_00-57.pt'
mymodel.load_state_dict(torch.load(path))
mymodel.eval()

while(True):
    idx = int(input(f'{len(test_indexs)}: '))
    print(f'data num: ',test_indexs[idx])
    if idx==-1:
        break
    input_graph = next(indicator_coordinates_graph([test_indexs[idx]],width=width,height=height,delta=delta,is_optimal=is_optimal))

    # input_graph2 = my_ae_model.enc(torch.tensor(input_graph,dtype=torch.float32)).tolist()
    input_graph2 = input_graph

    graph_start_goal_path = os.getcwd()
    if is_optimal:
        graph_start_goal_path += f'/data/graph_start_goal_path/optimal/graph_start_goal_path{test_indexs[idx]}_{delta}.npy'
    else:
        graph_start_goal_path += f'/data/graph_start_goal_path/non_optimal/graph_start_goal_path{test_indexs[idx]}_{delta}.npy'
    start_goal = np.load(graph_start_goal_path)[0:4].tolist()

    for sub_graph in input_graph2:
        sub_graph += start_goal
                
    if is_optimal:
        tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/optimal/obstacle_label{test_indexs[idx]}_{delta}.npy').tolist()
    else:
        tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/non_optimal/obstacle_label{test_indexs[idx]}_{delta}.npy').tolist()
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

    c = mymodel.embedding(tmp_data.x, tmp_data.edge_index)
    z = torch.randn(SAGEcVAE_config['z_dim']).to(device)
    z = z.repeat(c.size()[0],1)
    print(z.size())
    print(c.size())
    z = torch.cat([z,c],dim=1)
    generated_y = mymodel.decoder(z, tmp_data.edge_index)

    print('generation:',generated_y.view(-1))
    print('data:',tmp_data.y)

while True:
    data = connectionSock.recv(1024)
    print('received data : ', data.decode('utf-8'))
    connectionSock.send('I am a server.'.encode('utf-8'))
    print('send message.')