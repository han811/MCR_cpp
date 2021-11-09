import argparse
import logging
import os
from sys import path
from datetime import datetime
from typing import *

import numpy as np
from torch.serialization import save
from torch.types import Device
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data.data import Data
import torch.optim as optim
from torch_geometric.loader import DataLoader

from myGNN_preprocessing import train_validation_test_data_load, indicator_coordinates_graph, all_indicator_coordinates_graph
from myModel_config import AE_config
from myGNN_utils import get_n_params, TqdmLoggingHandler
from myModels import myAutoEncoder



'''
    This is train code for one-hot key configuration and indicator-configuration concatenated feature vector
'''

'''
    python3 myGNN_train_feature_ae.py --epochs 128 --batch_size 32 --train 1 --delta 0.1 --width 12 --height 12
'''

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--epochs', required=False, default=512, type=int, help='num of epochs')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')
parser.add_argument('--log_step', required=False, default=100, type=int, help='show the training log per each log step')
parser.add_argument('--save_step', required=False, default=100, type=int, help='setting model saving step')
parser.add_argument('--train', required=False, default=False, type=bool, help='whether train or evaluate')
parser.add_argument('--learning_rate', required=False, default=1e-2, type=float, help='learning rate of training')
parser.add_argument('--batch_size', required=False, default=32, type=int, help='setting batch size')

parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=12.0, type=float, help='height size of map')

parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--total', required=False, default=False, type=bool, help='choose total dataset or not')

parser.add_argument('--stop_iters', required=False, default=10, type=int, help='set minimum epoch number until model update')


args = parser.parse_args()

epochs = args.epochs
device = args.device
log_step = args.log_step
save_step = args.save_step
TRAIN = args.train
learning_rate = args.learning_rate
batch_size = args.batch_size

width = args.width
height = args.height

delta = args.delta
total = args.total

stop_iters = args.stop_iters

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()


#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_indexs, validation_indexs, test_indexs = train_validation_test_data_load()

in_node = len(next(indicator_coordinates_graph(train_indexs, width=width, height=height, delta=delta))[0])

print(f'train : {len(train_indexs)}\nvalidation : {len(validation_indexs)}\ntest : {len(test_indexs)}')
print()
print(f'node feature 0-1 size : {in_node}')
print()


#########################################################################################################################
'''  select which model to train  '''
#########################################################################################################################################
'''
    you have to set model layer parameters in myGNN_config.py file
'''
mymodel = myAutoEncoder(in_channels=in_node, **AE_config)
print(mymodel)
print()
print(f'number of clf model parameters: {get_n_params(mymodel)}')
print()


#########################################################################################################################################
'''  set training parameters  '''
#########################################################################################################################################
optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate)

# loss_function = nn.BCELoss()
loss_function = nn.MSELoss()

mymodel = mymodel.to(device)
loss_function = loss_function.to(device)

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
    train_loader = DataLoader(all_indicator_coordinates_graph(train_indexs, width=width, height=height), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(all_indicator_coordinates_graph(validation_indexs, width=width, height=height), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(all_indicator_coordinates_graph(test_indexs, width=width, height=height), batch_size=batch_size, shuffle=True)


#########################################################################################################################################
'''  start training  '''
#########################################################################################################################################
if TRAIN:
    # setting for save best model
    save_loss = 0
    pre_save_loss = 0

    tmp_stop_iters = 0

    for epoch in tqdm(range(epochs), desc='Epoch'):
        avg_loss = 0

        if total:
            for batch_idx, batch_data in tqdm(enumerate(train_loader), leave=False, desc='train'):
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                prediction_y = mymodel(batch_data.x)

                loss = loss_function(prediction_y,batch_data.x)
                    
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item()
                    
                if batch_idx % log_step == 0:
                    log.info(f'each batch loss: {loss.item():0.5f}')
        else:
            for idx in tqdm(range(0,len(train_indexs),batch_size), leave=False, desc='Batch'):
                current_train_indexs = train_indexs[idx*batch_size:(idx+1)*batch_size]
                current_data = []
                for idx2, input_graph in enumerate(indicator_coordinates_graph(current_train_indexs, width=width, height=height, delta=delta)):
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
                    current_data.append(Data(x=torch.tensor(input_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge)))
                if len(current_data)==batch_size:
                    current_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True)
                elif len(current_data)!=batch_size and len(current_data)>0:
                    current_loader = DataLoader(current_data, batch_size=len(current_data), shuffle=True)

                for batch_data in current_loader:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    prediction_y = mymodel(batch_data.x)

                    loss = loss_function(prediction_y,batch_data.x)

                    loss.backward()
                    optimizer.step()
                    
                    avg_loss += loss.item()
                    
                if idx % log_step == 0:
                    log.info(f'each batch loss: {loss.item():0.5f}')
        
        avg_loss /= len(range(0,len(train_indexs),batch_size))
        print(f'epoch loss: {avg_loss}')
        
        # test part
        test_avg_loss = 0

        with torch.no_grad():
            if total:
                for batch_idx, batch_data in tqdm(enumerate(test_loader), desc='test'):
                    n_obstacle = batch_data.x.size()[0]
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    prediction_y = mymodel(batch_data.x)

                    loss = loss_function(prediction_y,batch_data.x)

                    test_avg_loss += loss.item()
                    log.info(f'test each batch loss: {loss.item():0.5f}')
            else:
                for idx in tqdm(range(0,len(test_indexs),batch_size), leave=False, desc='Test Batch'):
                    current_test_indexs = test_indexs[idx*batch_size:(idx+1)*batch_size]
                    current_data = []
                    for idx2, input_graph in enumerate(indicator_coordinates_graph(current_test_indexs, width=width, height=height, delta=delta)):
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
                        current_data.append(Data(x=torch.tensor(input_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge)))
                    if len(current_data)==batch_size:
                        current_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True)
                    elif len(current_data)!=batch_size and len(current_data)>0:
                        current_loader = DataLoader(current_data, batch_size=len(current_data), shuffle=True)

                    for batch_data in current_loader:
                        batch_data = batch_data.to(device)
                        prediction_y = mymodel(batch_data.x)
                        
                        loss = loss_function(prediction_y,batch_data.x)

                        test_avg_loss += loss.item()
                        
                        # log.info(f'test each batch loss: {loss.item():0.5f}')
                        # log.info(f'test prediction: {prediction_y.view(-1)}')
                        # log.info(f'test label: {batch_data.x.cpu().view(-1)}')
            test_avg_loss /= len(range(0,len(test_indexs),batch_size))
            
            if pre_save_loss == 0.0:
                pre_save_loss = test_avg_loss
            else:
                save_loss = test_avg_loss
                if save_loss < pre_save_loss:
                    tmp_stop_iters = 0
                    pre_save_loss = save_loss
                    torch.save(mymodel.state_dict(), f'./save_model/AE/feature_ae_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
                else:
                    tmp_stop_iters += 1
                    if tmp_stop_iters > stop_iters:
                        break
            print(f'test epoch loss: {test_avg_loss}')
    torch.save(mymodel.state_dict(), f'./save_model/AE/feature_ae_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
else:
    path = f'./save_model/AE/feature_ae_sigmoid_mse_64_fixed_2layers_1024.pt'
    mymodel.load_state_dict(torch.load(path))
    mymodel.eval()
    while True:
        idx = int(input(f'{len(test_indexs)}: '))
        if idx==-1:
            break
        input_graph = next(indicator_coordinates_graph([test_indexs[idx]],width=width,height=height,delta=delta))
        input_graph = torch.tensor(input_graph,dtype=torch.float32)
        input_graph.to('cpu')
        mymodel.to('cpu')
        prediction_y = mymodel(input_graph)
        embedding_vector = mymodel.enc(input_graph)
        # print('prediction:',prediction_y[0].tolist())
        # print('true:',input_graph[0].tolist())

        print('embedding vector')
        print(embedding_vector[0].tolist())
        print(embedding_vector[0].size())
        print('prediction:',(prediction_y[0]<0.5).sum())
        print('true:',(input_graph[0]<0.5).sum())