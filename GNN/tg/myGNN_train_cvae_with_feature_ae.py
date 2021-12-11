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

parser.add_argument('--width', required=False, default=12.0, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8.0, type=float, help='height size of map')

parser.add_argument('--delta', required=False, default=0.1, type=float, help='key configuration granularity')
parser.add_argument('--total', required=False, default=False, type=bool, help='choose total dataset or not')

parser.add_argument('--stop_iters', required=False, default=10, type=int, help='set minimum epoch number until model update')

parser.add_argument('--is_optimal', required=False, default=False, type=bool, help='train model on optimal or non_optimal')

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

width = args.width
height = args.height

delta = args.delta
total = args.total

stop_iters = args.stop_iters

is_optimal = args.is_optimal

beta = 1

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()


#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_indexs, validation_indexs, test_indexs = train_validation_test_data_load(is_optimal=is_optimal)

# in_node = len(next(indicator_coordinates_graph([train_indexs[0]], delta=delta, is_optimal=is_optimal))[0])
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
optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate)

reconstruction_loss_function = nn.BCELoss(reduction='sum')
# reconstruction_loss_function = nn.MSELoss(reduction='sum')
# reconstruction_loss_function = nn.L1Loss(reduction='sum')

mymodel = mymodel.to(device)
reconstruction_loss_function = reconstruction_loss_function.to(device)

n_train_set = len(train_indexs)
n_validation_set = len(validation_indexs)
n_test_set = len(test_indexs)

if is_save_weight:
    with open(f'./weights/{model}/weights_initialization','wb') as f:
        pickle.dump(mymodel.state_dict(),f,pickle.HIGHEST_PROTOCOL)

if is_tensorboard:
    embedding_channels = SAGEcVAE_config['embedding_channels']
    en_hidden_channels = SAGEcVAE_config['en_hidden_channels']
    de_hidden_channels = SAGEcVAE_config['de_hidden_channels']
    z_dim = SAGEcVAE_config['z_dim']
    writer = SummaryWriter(f'GNN_model_train/{model}/{delta}_is_optimal_{is_optimal}_my_model_{model}_beta_{beta}_z_dim_{z_dim}_embedding_channels_{embedding_channels}_en_hidden_channels_{en_hidden_channels}_de_hidden_channels_{de_hidden_channels}_lr_{learning_rate}_batch_{batch_size}_activation_{activation}_parametes_{get_n_params(mymodel)}_ntrain_{len(train_indexs)}_nvalidation_{len(validation_indexs)}_ntest_{len(test_indexs)}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')

#########################################################################################################################################
'''  setting for logger  '''
#########################################################################################################################################
log = logging.getLogger(__name__)
log.handlers = []
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

if total:
    train_loader = DataLoader(all_indicator_coordinates_graph(train_indexs, is_optimal=is_optimal), batch_size=batch_size, shuffle=True, is_optimal=is_optimal)
    validation_loader = DataLoader(all_indicator_coordinates_graph(validation_indexs, is_optimal=is_optimal), batch_size=batch_size, shuffle=True, is_optimal=is_optimal)
    test_loader = DataLoader(all_indicator_coordinates_graph(test_indexs, is_optimal=is_optimal), batch_size=batch_size, shuffle=True, is_optimal=is_optimal)


#########################################################################################################################################
'''  start training  '''
#########################################################################################################################################
if TRAIN:
    current_accuracy = 0
    pre_accuracy = 0
    count_stop_iter = 0

    for epoch in tqdm(range(epochs), desc='Epoch'):
        '''
            set-up for calculate total
        '''
        avg_loss = 0
        avg_kl_loss = 0
        avg_reconstruction_loss = 0

        for idx in tqdm(range(0,len(train_indexs),batch_size), leave=False, desc='Batch'):
            current_train_indexs = train_indexs[idx:idx+batch_size]
            current_data = []
            for idx2, input_graph in enumerate(indicator_coordinates_graph(current_train_indexs,width=width,height=height,delta=delta,is_optimal=is_optimal)):
                graph_start_goal_path = os.getcwd()
                if is_optimal:
                    graph_start_goal_path += f'/data/graph_start_goal_path/optimal/graph_start_goal_path{current_train_indexs[idx2]}_{delta}.npy'
                else:
                    graph_start_goal_path += f'/data/graph_start_goal_path/non_optimal/graph_start_goal_path{current_train_indexs[idx2]}_{delta}.npy'
                start_goal = np.load(graph_start_goal_path)[0:4].tolist()

                input_graph2 = input_graph
                
                for sub_graph in input_graph2:
                    sub_graph += start_goal

                if is_optimal:
                    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/optimal/obstacle_label{current_train_indexs[idx2]}_{delta}.npy').tolist()
                else:
                    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/non_optimal/obstacle_label{current_train_indexs[idx2]}_{delta}.npy').tolist()
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
                current_data.append(Data(x=torch.tensor(input_graph2,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32)))
            if len(current_data)==batch_size:
                current_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True)
                tmp_batch_size = batch_size
            elif len(current_data)!=batch_size and len(current_data)>0:
                current_loader = DataLoader(current_data, batch_size=len(current_data), shuffle=True)
                tmp_batch_size = len(current_data)

            for batch_idx, batch_data in enumerate(current_loader):
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                if model=='SAGEcVAE':
                    reconstruction_y, z_mu, z_logvar = mymodel(batch_data.x,batch_data.edge_index,batch_data.y,tmp_batch_size)
                reconstruction_loss = reconstruction_loss_function(reconstruction_y.view(-1),batch_data.y)
                reconstruction_loss /= tmp_batch_size
                kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim = 1), dim = 0)
                loss = reconstruction_loss + kl_loss * beta
                
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * tmp_batch_size
                avg_kl_loss += kl_loss.item() * tmp_batch_size
                avg_reconstruction_loss += reconstruction_loss.item() * tmp_batch_size
                
            if idx % log_step == 0:
                log.info(f'each batch loss: {loss.item():0.5f}')
                log.info(f'each batch reconstruction_loss: {reconstruction_loss.item():0.5f}')
                log.info(f'each batch kl_loss: {kl_loss.item():0.5f}')
        
        avg_loss /= (n_train_set)
        avg_kl_loss /= (n_train_set)
        avg_reconstruction_loss /= (n_train_set)

        print(f'epoch avg_loss: {avg_loss}')
        print(f'epoch avg_reconstruction_loss: {avg_reconstruction_loss}')
        print(f'epoch avg_kl_loss: {avg_kl_loss}')

        if is_tensorboard:
            writer.add_scalar("my_GNN_epoch_loss", avg_loss, epoch)
            writer.add_scalar("my_GNN_epoch_reconstruction_loss", avg_reconstruction_loss, epoch)
            writer.add_scalar("my_GNN_epoch_kl_loss", avg_kl_loss, epoch)
            for name, param in mymodel.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch)
            for tag, value in mymodel.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)
        
        # test part
        test_avg_loss = 0
        test_avg_kl_loss = 0
        test_avg_reconstruction_loss = 0

        test_mode_accuracy = 0

        with torch.no_grad():
            current_data = []
            for data_idx, input_graph in enumerate(indicator_coordinates_graph(test_indexs,width=width,height=height,delta=delta,is_optimal=is_optimal)):
                graph_start_goal_path = os.getcwd()
                if is_optimal:
                    graph_start_goal_path += f'/data/graph_start_goal_path/optimal/graph_start_goal_path{test_indexs[data_idx]}_{delta}.npy'
                else:
                    graph_start_goal_path += f'/data/graph_start_goal_path/non_optimal/graph_start_goal_path{test_indexs[data_idx]}_{delta}.npy'
                start_goal = np.load(graph_start_goal_path)[0:4].tolist()

                input_graph2 = input_graph

                for sub_graph in input_graph2:
                    sub_graph += start_goal

                if is_optimal:
                    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/optimal/obstacle_label{test_indexs[data_idx]}_{delta}.npy').tolist()
                else:
                    tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/non_optimal/obstacle_label{test_indexs[data_idx]}_{delta}.npy').tolist()
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
                current_data.append(Data(x=torch.tensor(input_graph2,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32)))
            current_loader = DataLoader(current_data, batch_size=n_test_set, shuffle=True)

            for batch_data in current_loader:
                batch_data = batch_data.to(device)
                tmp_batch_size = n_test_set
                if model=='SAGEcVAE':
                    reconstruction_y, z_mu, z_logvar = mymodel(batch_data.x,batch_data.edge_index,batch_data.y,tmp_batch_size)
                reconstruction_loss = reconstruction_loss_function(reconstruction_y.view(-1),batch_data.y)
                reconstruction_loss /= n_test_set
                kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim = 1), dim = 0)
                loss = reconstruction_loss + kl_loss * beta


                test_avg_loss += loss.item() * tmp_batch_size
                test_avg_kl_loss += kl_loss.item() * tmp_batch_size
                test_avg_reconstruction_loss += reconstruction_loss.item() * tmp_batch_size
                

                c = mymodel.embedding(batch_data.x, batch_data.edge_index)
                z = torch.randn(tmp_batch_size, SAGEcVAE_config['z_dim']).to(device)
                repeat_num = int(c.size()[0]/tmp_batch_size)
                for tmp_idx, tmp_z in enumerate(z):
                    if tmp_idx==0:
                        z_next = tmp_z.unsqueeze(0)
                        z_next = z_next.repeat(repeat_num,1)
                    else:
                        tmp_z_next = tmp_z.unsqueeze(0)
                        tmp_z_next = tmp_z_next.repeat(repeat_num,1)
                        z_next = torch.cat([z_next,tmp_z_next], dim=0)
                z = torch.cat([z_next,c],dim=1)
                generated_y = mymodel.decoder(z, batch_data.edge_index)
                label_y = torch.reshape(batch_data.y, (-1,n_obstacle))
                generated_y = torch.reshape(generated_y, (-1,n_obstacle))
                for test_index in range(tmp_batch_size):
                    tmp_y = label_y[test_index]
                    tmp_generated_y_0 = (generated_y[test_index] >= probabilistic_threshold)
                    tmp_generated_y_1 = (generated_y[test_index] < probabilistic_threshold)
                    loss0 = nn.MSELoss()(tmp_y, tmp_generated_y_0)
                    loss1 = nn.MSELoss()(tmp_y, tmp_generated_y_1)
                    if loss0 and loss1:
                        pass
                    else:
                        test_mode_accuracy += 1

                log.info(f'test each batch loss: {loss.item():0.5f}')
                log.info(f'test each batch kl loss: {kl_loss.item():0.5f}')
                log.info(f'test each batch reconstruction loss: {reconstruction_loss.item():0.5f}')
            
            test_avg_loss /= (n_test_set)
            test_avg_kl_loss /= (n_test_set)
            test_avg_reconstruction_loss /= (n_test_set)
            test_mode_accuracy /= (n_test_set)

            print(f'test epoch loss: {test_avg_loss}')
            print(f'test epoch kl loss: {test_avg_kl_loss}')
            print(f'test epoch reconstruction loss: {test_avg_reconstruction_loss}')
            # print(f'test epoch mode accuracy: {test_mode_accuracy}')
            
        if is_tensorboard:
            writer.add_scalar("test_epoch_loss", test_avg_loss, epoch)
            writer.add_scalar("test_epoch_kl_loss", test_avg_kl_loss, epoch)
            writer.add_scalar("test_epoch_reconstruction_loss", test_avg_reconstruction_loss, epoch)
        
        if epoch==0:
            current_accuracy = test_avg_reconstruction_loss
        else:
            pre_accuracy = current_accuracy
            current_accuracy = test_avg_reconstruction_loss

        if current_accuracy > pre_accuracy:
            embedding_channels = SAGEcVAE_config['embedding_channels']
            en_hidden_channels = SAGEcVAE_config['en_hidden_channels']
            de_hidden_channels = SAGEcVAE_config['de_hidden_channels']
            z_dim = SAGEcVAE_config['z_dim']
            torch.save(mymodel.state_dict(), f'./save_model/{model}/{delta}_is_optimal_{is_optimal}_my_model_{model}_beta_{beta}_z_dim_{z_dim}_embedding_channels_{embedding_channels}_en_hidden_channels_{en_hidden_channels}_de_hidden_channels_{de_hidden_channels}_lr_{learning_rate}_batch_{batch_size}_activation_{activation}_parametes_{get_n_params(mymodel)}_ntrain_{len(train_indexs)}_nvalidation_{len(validation_indexs)}_ntest_{len(test_indexs)}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')

        if current_accuracy:
            if current_accuracy > pre_accuracy:
                count_stop_iter = 0
            else:
                count_stop_iter += 1
            
            if count_stop_iter > stop_iters:
                break
        '''
            don't forget rename the model after training!!!!!
        '''
        if is_save_weight:
            with open(f'./weights/{model}/weights_{epoch}','wb') as f:
                pickle.dump(mymodel.state_dict(),f,pickle.HIGHEST_PROTOCOL)
    torch.save(mymodel.state_dict(), f'./save_model/{model}/final_{delta}_is_optimal_{is_optimal}_my_model_{model}_beta_{beta}_z_dim_{z_dim}_embedding_channels_{embedding_channels}_en_hidden_channels_{en_hidden_channels}_de_hidden_channels_{de_hidden_channels}_lr_{learning_rate}_batch_{batch_size}_activation_{activation}_parametes_{get_n_params(mymodel)}_ntrain_{len(train_indexs)}_nvalidation_{len(validation_indexs)}_ntest_{len(test_indexs)}_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')

    mymodel.eval()
    while(True):
        idx = int(input(f'{len(test_indexs)}: '))
        if idx==-1:
            break
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
        tmp_data.to(device)
        c = mymodel.embedding(tmp_data.x, tmp_data.edge_index)
        print(c.size())
        z = torch.randn(SAGEcVAE_config['z_dim']).to(device)
        z = z.repeat(c.size()[0],1)
        z = torch.cat([z,c],dim=1)
        generated_y = mymodel.decoder(z, tmp_data.edge_index)

        print('generation:',generated_y.view(-1))
        print('data:',tmp_data.y)
else:
    path = './save_model/SAGEcVAE/0.2_is_optimal_False_my_model_SAGEcVAE_beta_1_z_dim_8_embedding_channels_4_en_hidden_channels_64_de_hidden_channels_4_lr_0.01_batch_128_activation_elu_parametes_13521_ntrain_19279_nvalidation_0_ntest_2143_2021-12-08_14-04.pt'
    mymodel.load_state_dict(torch.load(path))
    mymodel.eval()

    test_fail_cases = []
    test_fail_indexes = []
    test_avg_loss = 0
    test_avg_kl_loss = 0
    test_avg_reconstruction_loss = 0

    test_mode_accuracy = 0
    
    while(True):
        idx = int(input(f'{len(test_indexs)}: '))
        print(f'data num: ',test_indexs[idx])
        if idx==-1:
            break
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
        tmp_data.to(device)

        c = mymodel.embedding(tmp_data.x, tmp_data.edge_index)
        z = torch.randn(SAGEcVAE_config['z_dim']).to(device)
        z = z.repeat(c.size()[0],1)
        print(z.size())
        print(c.size())
        z = torch.cat([z,c],dim=1)
        generated_y = mymodel.decoder(z, tmp_data.edge_index)

        print('generation:',generated_y.view(-1)>0.5)
        print('data:',tmp_data.y)
