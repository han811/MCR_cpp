from datetime import datetime
from typing import *
import argparse
import random
import pickle
import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader

from tg_Preprocessing import load_tg_data
from myGNNs import myGCN, myGraphSAGE, myGAT, myGATcVAE
from myGNN_config import GCN_config, SAGE_config, GAT_config, GATcVAE_config
from myGNN_utils import FocalLoss

#########################################################################################################################################
'''  setting training parameters  '''
#########################################################################################################################################
parser = argparse.ArgumentParser(description='Training my GNN models')

parser.add_argument('--model', required=True, default='GCN', type=str, help='choose model among GCN, SAGE, GAT, GATcVAE')
parser.add_argument('--epochs', required=False, default=512, type=int, help='num of epochs')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')
parser.add_argument('--log_step', required=False, default=100, type=int, help='show the training log per each log step')
parser.add_argument('--save_step', required=False, default=100, type=int, help='setting model saving step')
parser.add_argument('--train', required=False, default=True, type=bool, help='whether train or evaluate')
parser.add_argument('--learning_rate', required=False, default=1e-2, type=float, help='learning rate of training')
parser.add_argument('--probabilistic_threshold', required=False, default=0.5, type=float, help='setting probabilistic threshold')
parser.add_argument('--batch_size', required=False, default=32, type=int, help='setting batch size')
parser.add_argument('--data_size', required=False, default=3, type=int, help='setting data size')

parser.add_argument('--tensorboard', required=False, default=True, type=bool, help='record on tensorboard or not')
parser.add_argument('--is_save_weight', required=False, default=False, type=bool, help='record weights or not')
parser.add_argument('--is_save_hidden', required=False, default=False, type=bool, help='record hiddens or not')

parser.add_argument('--problem_type', required=True, default='clf', type=str, help='setting problem type')

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
data_size = (0,args.data_size)

is_tensorboard = args.tensorboard
is_save_weight = args.is_save_weight
is_save_hidden = args.is_save_hidden

problem_type = args.problem_type

for arg in vars(args):
    indent = 28 - len(arg)
    print(f'{arg}  {getattr(args, arg):>{indent}}')
print()

#########################################################################################################################################
'''  load train validation test dataset  '''
#########################################################################################################################################
train_set:list = []
val_set:list = []
test_set:list = []
for data_num in range(data_size[0],data_size[1]):
    tmp_train_set, tmp_val_set, tmp_test_set = load_tg_data(num=data_num)
    train_set += tmp_train_set
    val_set += tmp_val_set
    test_set += tmp_test_set

in_node = len(train_set[0].x[0])

index = []
for idx,t in enumerate(train_set):
    if t.x.size()[1]!=in_node:
        index.append(idx)
index.reverse()
for t in index:
    del train_set[t]

index = []
for idx,t in enumerate(val_set):
    if t.x.size()[1]!=in_node:
        index.append(idx)
index.reverse()
for t in index:
    del val_set[t]

index = []
for idx,t in enumerate(test_set):
    if t.x.size()[1]!=in_node:
        index.append(idx)
index.reverse()
for t in index:
    del test_set[t]

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(f'train : {len(train_set)}\nvalidation : {len(val_set)}\ntest : {len(test_set)}')
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
elif model=='GAT':
    mymodel = myGAT(in_channels=in_node, is_save_hiddens=is_save_hidden, **GAT_config)
elif model=='SAGEcVAE':
    pass
elif model=='GATcVAE':
    mymodel = myGATcVAE(en_in_channels=in_node, de_in_channels=in_node, is_save_hiddens=is_save_hidden, **GATcVAE_config)
    z_dim = GATcVAE_config['z_dim']
print(mymodel)
print()
num_param = 0
for p in list(mymodel.parameters()):
    tmp_p = 1
    for s in list(p.size()):
        tmp_p = tmp_p*s
    num_param += tmp_p
print(f'number of model parameters: {num_param}')
print()

#########################################################################################################################################
'''  set training parameters  '''
#########################################################################################################################################
# optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate, weight_decay=0.9)
optimizer = optim.Adam(mymodel.parameters(),lr=learning_rate)

if model=='GCN' or model=='SAGE' or model=='GAT':
    # loss_function = nn.BCELoss()
    loss_function = FocalLoss(gamma=6.5,alpha=5)
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

n_train_loader = len(train_loader)
n_train_set = len(train_set)
n_val_loader = len(val_loader)
n_val_set = len(val_set)
n_test_loader = len(test_loader)
n_test_set = len(test_set)
n_obstacles = len(train_set[0].x)


#########################################################################################################################################
'''  start training  '''
#########################################################################################################################################
if TRAIN:
    # write model in graph
    if problem_type=='clf':
        writer.add_graph(mymodel, (train_set[0].x.to(device), train_set[0].edge_index.to(device)))
    elif problem_type=='generation':
        writer.add_graph(mymodel, (train_set[0].x.to(device), train_set[0].edge_index.to(device), train_set[0].y.to(device)))

    # setting for save best model
    pre_accuracy = 0
    current_accuracy = 0

    for epoch in range(epochs):
        print(f'epoch: {epoch+1}')
        
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

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            if model=='GCN' or model=='SAGE':
                prediction_y = mymodel(data.x,data.edge_index)
            elif model=='GAT':
                prediction_y, attentions = mymodel(data.x,data.edge_index)
            elif model=='SAGEcVAE' or model=='GATcVAE':
                reconstruction_y, z_mu, z_log_var = mymodel(data.x,data.edge_index,data.y)
            
            if problem_type=='clf':
                loss = loss_function(prediction_y,data.y.unsqueeze(1))
            elif problem_type=='generation':
                mse = loss_function(reconstruction_y,data.y.unsqueeze(1))
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
                avg_accuracy += (prediction_y_acc == data.y.unsqueeze(1)).sum().item()
                count_0 += data.y.size()[0] - data.y.count_nonzero()
                count_1 += data.y.count_nonzero()
                for index, tmp_y in enumerate(data.y):
                    if tmp_y.item()==0:
                        if tmp_y==prediction_y_acc[index]:
                            avg_0_accuracy += 1
                    elif tmp_y.item()==1:
                        if tmp_y==prediction_y_acc[index]:
                            avg_1_accuracy += 1
                print(f'each batch: {loss.item()}',end='\r')
            elif problem_type=='generation':
                print(f'each batch loss: {loss.item():0.5f}    each batch mse: {mse.item():0.5f}    each batch kl: {kl:0.5f}',end='\r')


        if problem_type=='clf':
            avg_loss /= n_train_loader
            avg_accuracy /= (n_train_set*n_obstacles)
            avg_0_accuracy /= count_0
            avg_1_accuracy /= count_1
            if epoch % log_step == 0:
                print(f'epoch loss: {avg_loss}')
                print(f'epoch accuracy: {avg_accuracy*100}%')
                print(f'epoch 0 accuracy: {avg_0_accuracy*100}%')
                print(f'epoch 1 accuracy: {avg_1_accuracy*100}%')
            writer.add_scalar("my_GNN_epoch_loss", avg_loss, epoch)
            writer.add_scalar("my_GNN_epoch_accuracy", avg_accuracy, epoch)
            writer.add_scalar("my_GNN_epoch_0_accuracy", avg_0_accuracy, epoch)
            writer.add_scalar("my_GNN_epoch_1_accuracy", avg_1_accuracy, epoch)
        elif problem_type=='generation':
            avg_loss /= n_train_loader
            avg_mse /= n_train_loader
            avg_kl /= n_train_loader
            if epoch % log_step == 0:
                print()
                print(f'epoch loss: {avg_loss}')
                print(f'epoch mse: {avg_mse}')
                print(f'epoch kl: {avg_kl}')
            writer.add_scalar("my_GNN_epoch_loss", avg_loss, epoch)
            writer.add_scalar("my_GNN_epoch_mse", avg_mse, epoch)
            writer.add_scalar("my_GNN_epoch_kl", avg_kl, epoch)

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
            random_sample = random.randint(0,n_test_loader)
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                if model=='GCN' or model=='SAGE':
                    prediction_y = mymodel(data.x,data.edge_index)
                elif model=='GAT':
                    prediction_y, attentions = mymodel(data.x,data.edge_index)
                elif model=='SAGEcVAE' or model=='GATcVAE':
                    z = torch.randn(z_dim).to(device)
                    reconstruction_y = mymodel.decoder(data.x,data.edge_index,z)

                if problem_type == 'clf':
                    loss = loss_function(prediction_y,data.y.unsqueeze(1))
                    test_avg_loss += loss.item()
                    prediction_y_acc = prediction_y>probabilistic_threshold
                    test_avg_accuracy += (prediction_y_acc == data.y.unsqueeze(1)).sum().item()
                    test_count_0 += data.y.size()[0] - data.y.count_nonzero()
                    test_count_1 += data.y.count_nonzero()
                    for index, tmp_y in enumerate(data.y):
                        if tmp_y.item()==0:
                            if tmp_y==prediction_y_acc[index]:
                                test_avg_0_accuracy += 1
                        elif tmp_y.item()==1:
                            if tmp_y==prediction_y_acc[index]:
                                test_avg_1_accuracy += 1
                    print(f'test each batch: {loss.item()}',end='\r')
                elif problem_type == 'generation':
                    if random_sample==batch_idx:
                        print(reconstruction_y.view(-1))
                        print(data.y.view(-1))
            
            if problem_type == 'clf':
                test_avg_loss /= n_test_loader
                test_avg_accuracy /= (n_test_set*n_obstacles)
                test_avg_0_accuracy /= test_count_0
                test_avg_1_accuracy /= test_count_1
                if epoch % log_step == 0:
                    print(f'test epoch loss: {test_avg_loss}')
                    print(f'test epoch accuracy: {test_avg_accuracy*100}%')
                    print(f'test epoch 0 accuracy: {test_avg_0_accuracy*100}%')
                    print(f'test epoch 1 accuracy: {test_avg_1_accuracy*100}%')
 
                writer.add_scalar("test_epoch_loss", test_avg_loss, epoch)
                writer.add_scalar("test_epoch_accuracy", test_avg_accuracy, epoch)
                writer.add_scalar("test_epoch_0_accuracy", test_avg_0_accuracy, epoch)
                writer.add_scalar("test_epoch_1_accuracy", test_avg_1_accuracy, epoch)

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
