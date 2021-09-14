import os, sys
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

from GNNVAE import VAE_Encoder, VAE_Decoder

from model import Node_Update_Function, Edge_to_Node_Aggregation_Function, GNN
from graph import Graph
from utils import graph_generate_load, init_weights, graph_generate, count_label_01, average_precision_recall_plot, key_configuration_load
from test import plot_with_labels
from config import model_config

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
PREPARE = model_config['PREPARE']
z_dim = model_config(['dim_z'])


if __name__=='__main__':
    # make a model
    print("Data preprocessing start!!")
    if PREPARE:
        graph_inputs = Graph(cuda=CUDA)
        train_data = graph_generate_load()
        print(train_data)
        exit()
        tmp_data = np.load("./train_data.npy",allow_pickle=True)
        tmp_x = tmp_data[0]
        tmp_edge = tmp_data[1]
        tmp_y = tmp_data[2]
        num_node = len(tmp_x[0])
        in_node = len(key_configuration_load())
        for idx in range(len(tmp_y)):
            graph_inputs.add_graph(list(tmp_x[idx]),list(tmp_edge[idx]),list(tmp_y[idx]))
        labels = tmp_y
    else:
        graph_inputs, in_node, labels = graph_generate(CUDA)
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
    
    generator = GAN_GNN_Generator(in_node,z_dim,[64,128,64])
    discriminator = GAN_GNN_Discriminator(num_node, in_node, [64,128,64])

    # define loss & optimizer
    # optimizer = optim.Adam(GNN_model.parameters(), lr=learning_rate)
    # weights = count_label_01()
    # class_weights = torch.FloatTensor(weights).cuda()
    # loss = nn.BCELoss(weight=(class_weights[0]/(class_weights[0]+class_weights[1])))
    loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()
        
    ############# fix for graph #############
    # train
    if TRAIN:
        for tmp_data in train_loader:
            x, l, adjancy_matrix = tmp_data[0], tmp_data[1], tmp_data[2]
            # # Adversarial ground truths
            valid = Variable(torch.Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

            # # Configure input
            # real_imgs = Variable(imgs.type(Tensor))

            # # -----------------
            # #  Train Generator
            # # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.Tensor(np.random.normal(0, 1, (z_dim))))

            # Generate a batch of images
            gen_label = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss(discriminator(gen_label), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(x), valid)
            fake_loss = loss(discriminator(gen_label.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()