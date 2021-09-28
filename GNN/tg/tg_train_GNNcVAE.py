from datetime import datetime
import os
import pickle
from typing import *

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torch.optim as optim
from tensorboardX import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tg_utils import get_n_params, init_weights, count_label_01_dataset, weighted_binary_cross_entropy, FocalLoss
from tg_config import tg_clf_config, tg_cVAE_config
from tg_Preprocessing import load_tg_data

from tg_GNNcVAE import GraphcVAE



# setting parameters
epochs = tg_cVAE_config['epochs']
learning_rate = tg_cVAE_config['learning_rate']
CUDA = tg_cVAE_config['CUDA']
log_step = tg_cVAE_config['log_step']
save_step = tg_cVAE_config['save_step']
TRAIN = tg_cVAE_config['TRAIN']
plot_mAP = tg_cVAE_config['plot_mAP']
probability_threshold = tg_cVAE_config['probability_threshold']
model_path = tg_cVAE_config['model_path']
batch_size = tg_cVAE_config['batch_size']
n_encoding_feature = tg_cVAE_config['n_encoding_feature']
in_node = tg_cVAE_config['in_node']
data_size = tg_cVAE_config['tg_clf_config']
beta = tg_cVAE_config['beta']


if __name__=='__main__':
    print("Data preprocessing start!!")
    print(in_node)
    print("------------------------")
    print("Data preprocessing end!!")

    train_set:list = []
    val_set:list = []
    test_set:list = []
    for i in range(data_size[0],data_size[1]):
        tmp_train_set, tmp_val_set, tmp_test_set = load_tg_data(num=i)
        train_set += tmp_train_set
        val_set += tmp_val_set
        test_set += tmp_test_set

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    GNN_cVAE = GraphcVAE(
     en_graph_in_channels=in_node, en_graph_hidden_channels=64, en_graph_num_layers=1, en_graph_out_channels=32,
     hidden_channels=32, num_layers=1, z_dim=32,
     de_graph_in_channels=in_node, de_graph_hidden_channels=256, de_graph_num_layers=1, de_graph_out_channels=128,
     num_z_layers=2,
     clf_num_layers=2,
     dropout=0.4, act=nn.ReLU(), graph_linear_num_layers=2)# clf_model.apply(init_weights)
    print(GNN_cVAE)
    

    optimizer = optim.Adam(GNN_cVAE.parameters(),lr=learning_rate, betas=(0.5, 0.999), weight_decay=5e-4)
    mse_loss = nn.MSELoss()

    # loss_function = weighted_binary_cross_entropy
    # loss_function = FocalLoss(gamma=4,alpha=1e0)

    if CUDA:
        GNN_cVAE.cuda()
        mse_loss.cuda()

    writer = SummaryWriter()

    n_train_loader = len(train_loader)
    n_train_set = len(train_set)
    n_obstacles = 25
    n_test_loader = len(test_loader)
    n_test_set = len(test_set)

    if TRAIN:
        pre_accuracy = 0
        current_accuracy = 0
        # draw graph
        tmp = train_set[0]
        tmp = tmp.cuda()
        writer.add_graph(GNN_cVAE, (tmp.x, tmp.edge_index, tmp.y))
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}')
            avg_loss = 0
            avg_kl = 0
            avg_mse = 0
            for batch_idx, data in enumerate(train_loader):
                # data = train_set[0]
                if CUDA:
                    data = data.cuda()
                optimizer.zero_grad()
                l, z_mu, z_log_var = GNN_cVAE(data.x,data.edge_index,data.y)
                kl = -0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim = 0)
                mse = mse_loss(l, data.y.unsqueeze(-1)) * 10
                loss = kl * beta + mse
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                avg_kl += kl.item()
                avg_mse += mse.item()
                if batch_idx % 1000 == 0:
                    print(l.view(-1))
                    print(data.y.view(-1))
                    print(z_mu.view(-1))
                    print(z_log_var.view(-1))
                print(f'each batch - kl:{kl.item()} mse:{mse.item()}',end='\r')
            avg_loss /= n_train_loader
            avg_kl /= n_train_loader
            avg_mse /= n_train_loader
            print(f'epoch loss: {avg_loss}')
            writer.add_scalar("tg_epoch_loss", avg_loss, epoch)
            writer.add_scalar("tg_epoch_kl", avg_kl, epoch)
            writer.add_scalar("tg_epoch_mse", avg_mse, epoch)

            for tag, value in GNN_cVAE.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)
            
            for name, param in GNN_cVAE.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch)
            with open(f'./weights/weights_{epoch}','wb') as f:
                pickle.dump(GNN_cVAE.state_dict(),f,pickle.HIGHEST_PROTOCOL)
        torch.save(GNN_cVAE.state_dict(), f'./save_model/tg_GNN_cVAE_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')