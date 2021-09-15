import os, sys
from datetime import datetime
from typing import List, Tuple, Dict

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

from data_class import MCRdata, GraphSaveClass, MyGraphSaveClass
from GNNVAE import VAE_Encoder, VAE_Decoder, reparameterize, loss_function

from model import Node_Update_Function, Edge_to_Node_Aggregation_Function, GNN
from graph import Graph
from utils import graph_generate_load, init_weights, count_label_01, key_configuration_load
from test import plot_with_labels
from vae_config import model_config

# setting parameters
epochs = model_config['epochs']
learning_rate = model_config['learning_rate']
CUDA = model_config['CUDA']
log_step = model_config['log_step']
save_step = model_config['save_step']
TRAIN = model_config['TRAIN']
plot_mAP = model_config['plot_mAP']
probability_threshold = model_config['probability_threshold']
model_path = model_config['model_path']
z_dim = model_config['dim_z']


if __name__=='__main__':
    # make a model
    print("Data preprocessing start!!")
    graph_inputs = Graph(cuda=CUDA)
    graph_inputs, in_node, labels = graph_generate_load()
    in_node += 4
    print("------------------------")
    print("Data preprocessing end!!")

    # make a dataset      
    train_length = int(len(graph_inputs) * 0.7)
    train_set, val_test_set, train_label, val_test_label = train_test_split(graph_inputs, labels, test_size=0.3)
    
    val_length = int((len(graph_inputs) - train_length) * 2 / 3)
    test_length = len(graph_inputs) - train_length - val_length
    val_set, test_set, val_label, test_label = train_test_split(val_test_set, val_test_label, test_size=0.33)

    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)
    
    encoder = VAE_Encoder(in_node, 32, dim_z=z_dim, activation='elu')
    decoder = VAE_Decoder(in_node, 32, dim_z=z_dim, activation='elu')

    # define loss & optimizer
    # optimizer = optim.Adam(GNN_model.parameters(), lr=learning_rate)
    # weights = count_label_01()
    # class_weights = torch.FloatTensor(weights).cuda()
    # loss = nn.BCELoss(weight=(class_weights[0]/(class_weights[0]+class_weights[1])))

    reconstruction_function = nn.MSELoss()

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if CUDA:
        reconstruction_function.cuda()
        encoder.cuda()
        decoder.cuda()

    writer = SummaryWriter()
    ############# fix for graph #############
    # train
    if TRAIN:
        for epoch in range(1,epochs+1):
            avg_recon_loss = 0
            avg_kld_loss = 0
            avg_loss_value = 0
            for step, tmp_data in enumerate(train_loader):
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                
                X, A, y = tmp_data[0], tmp_data[1], tmp_data[2]
                mu_logvar = encoder(X,A,y)
                mu = mu_logvar[0][:z_dim]
                logvar = mu_logvar[0][z_dim:]

                sample_z = reparameterize(mu,logvar,z_dim)

                recon_y = decoder(X,A,sample_z)
                
                recon_loss, kld_loss = loss_function(recon_y,y,mu,logvar)
                loss_value = recon_loss+kld_loss
                loss_value.backward()
        
                decoder_optimizer.step()
                encoder_optimizer.step()

                avg_recon_loss += recon_loss.item()
                avg_kld_loss += kld_loss.item()
                avg_loss_value += loss_value.item()

                writer.add_scalar("Each_Step_Loss/recon_loss", recon_loss.item(), step+(epoch-1)*len(train_loader))
                writer.add_scalar("Each_Step_Loss/kld_loss", kld_loss.item(), step+(epoch-1)*len(train_loader))
                writer.add_scalar("Each_Step_Loss/loss_value", loss_value.item(), step+(epoch-1)*len(train_loader))

                print('recon_loss',recon_loss.item())
                print('kld_loss',kld_loss.item())
                print('loss_value',loss_value.item())
                print()


            writer.add_scalar("Loss/recon_loss", avg_recon_loss/len(train_loader), epoch)
            writer.add_scalar("Loss/kld_loss", avg_kld_loss/len(train_loader), epoch)
            writer.add_scalar("Loss/loss_value", avg_loss_value/len(train_loader), epoch)