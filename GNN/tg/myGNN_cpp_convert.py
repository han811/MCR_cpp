import argparse
import logging
import os
import pickle
from sys import path
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
    This is train code for one-hot key configuration and indicator-configuration concatenated feature vector
'''

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--model', required=True, default='SAGEcVAE', type=str, help='choose model among GCN, SAGE, GAT, GATcVAE, SAGEcVAE')
parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8.0, type=float, help='height size of map')
parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--is_optimal', required=False, default=False, type=bool, help='train model on optimal or non_optimal')


args = parser.parse_args()

model = args.model
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
train_indexs, validation_indexs, test_indexs = train_validation_test_data_load(is_optimal=is_optimal)

in_node = len(next(indicator_coordinates_graph([test_indexs[0]], delta=delta, is_optimal=is_optimal))[0])
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
path = './save_model/SAGEcVAE/0.2_is_optimal_False_my_model_SAGEcVAE_beta_1_z_dim_8_embedding_channels_4_en_hidden_channels_64_de_hidden_channels_4_lr_0.01_batch_64_activation_elu_parametes_18585_ntrain_5457_nvalidation_0_ntest_607_2021-12-11_15-24.pt'
mymodel.load_state_dict(torch.load(path))
mymodel.eval()

idx = 0
input_graph = next(indicator_coordinates_graph([test_indexs[idx]],width=width,height=height,delta=delta,is_optimal=is_optimal))

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

c = mymodel.embedding(tmp_data.x, tmp_data.edge_index)
z = torch.randn(SAGEcVAE_config['z_dim'])
z = z.repeat(c.size()[0],1)
z = torch.cat([z,c],dim=1)
generated_y = mymodel.decoder(z, tmp_data.edge_index)


traced_script_embedding = torch.jit.trace(mymodel.embedding, (tmp_data.x, tmp_data.edge_index))
traced_script_decoder = torch.jit.trace(mymodel.decoder, (z, tmp_data.edge_index))

traced_script_embedding.save("embedding.pt")
traced_script_decoder.save("decoder.pt")