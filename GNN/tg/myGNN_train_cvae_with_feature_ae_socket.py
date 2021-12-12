import argparse
import os
from sys import path
from socket import *
from threading import currentThread
from typing import *
from time import sleep

import numpy as np

import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from myGNN_preprocessing import train_validation_test_data_load, indicator_coordinates_graph, obstacle_graph_processing_socket
from myGNNs import mySAGEcVAE
from myGNN_config import SAGEcVAE_config
from myGNN_utils import get_n_params


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

path = './save_model/SAGEcVAE/0.2_is_optimal_False_my_model_SAGEcVAE_beta_1_z_dim_8_embedding_channels_4_en_hidden_channels_64_de_hidden_channels_4_lr_0.01_batch_64_activation_elu_parametes_18585_ntrain_5457_nvalidation_0_ntest_607_2021-12-11_15-24.pt'
mymodel.load_state_dict(torch.load(path))
mymodel.eval()

while(True):
    data = connectionSock.recv(1024)
    if not data:
        print("no data")
        sleep(0.5)
        continue
    obstacles = data.decode('utf-8')
    obstacles = list(map(float,obstacles.split()))
    start = obstacles[:2]
    goal = obstacles[2:4]
    obstacle_graph = [[] for _ in range(int((len(obstacles[4:])-1)/2))]
    for i in range(len(obstacles[4:])-1):
        obstacle_graph[int(i/2)].append(obstacles[4:][i])
    obstacle_radius = obstacles[-1]
    print("n_obstacles:",len(obstacle_graph))
    graph, edges = obstacle_graph_processing_socket(start, goal, obstacle_graph, obstacle_radius, width=width, height=height, delta=delta, is_optimal=is_optimal)
    current_data = Data(x=torch.tensor(graph,dtype=torch.float32), edge_index=torch.LongTensor(edges))
    current_data.to(device)
    c = mymodel.embedding(current_data.x, current_data.edge_index)
    print(c.size())
    z = torch.randn(SAGEcVAE_config['z_dim']).to(device)
    z = z.repeat(c.size()[0],1)
    z = torch.cat([z,c],dim=1)
    generated_y = mymodel.decoder(z, current_data.edge_index)
    generated_y = list(map(str,list(map(int,(generated_y.view(-1)>probabilistic_threshold).tolist()))))
    print('generation:',generated_y)
    print('received data : ', obstacle_graph)
    connectionSock.send(''.join(generated_y).encode('utf-8'))
    print('send message.')
