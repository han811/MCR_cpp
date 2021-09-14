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

from GNNGAN import GNN_Generator, GNN_Discriminator
from graph import Graph
from utils import init_weights, graph_generate, count_label_01, average_precision_recall_plot, key_configuration_load
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


if __name__=='__main__':
    # make a model
    print("Data preprocessing start!!")
    if PREPARE:
        graph_inputs = Graph(cuda=CUDA)
        tmp_data = np.load("./train_data.npy",allow_pickle=True)
        tmp_x = tmp_data[0]
        tmp_edge = tmp_data[1]
        tmp_y = tmp_data[2]
        in_node = len(key_configuration_load())
        for idx in range(len(tmp_y)):
            graph_inputs.add_graph(list(tmp_x[idx]),list(tmp_edge[idx]),list(tmp_y[idx]))
        labels = tmp_y
    else:
        graph_inputs, in_node, labels = graph_generate(CUDA)

    generator = GNN_Generator(in_node,256,activation='elu')
    discriminator = GNN_Discriminator(in_node,256,activation='elu')
    
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    adversarial_criterion = nn.MSELoss()

    real_label = torch.full((1, 1), 1, dtype=torch.float32)
    fake_label = torch.full((1, 1), 0, dtype=torch.float32)



    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_criterion.cuda()
        real_label = real_label.cuda()
        fake_label = fake_label.cuda()

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
    
    # define loss & optimizer
    generator_opt = torch.optim.Adam(generator.parameters(), lr=learning_rate*10)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    # weights = count_label_01()
    # class_weights = torch.FloatTensor(weights).cuda()
    # loss = nn.BCELoss(weight=(class_weights[0]/(class_weights[0]+class_weights[1])))
    # loss = nn.BCELoss()

    # if CUDA:
    #     GNN_model.cuda()
    #     loss.cuda()
        
    # train
    if TRAIN:
        for epoch in range(epochs):
            total_g_loss = 0
            total_d_loss = 0

            train_n = 0
            val_n = 0

            for graph_, edge_, label_ in train_loader:
                discriminator.zero_grad()
                real_output = discriminator(graph_,edge_,label_)
                d_loss_real = adversarial_criterion(real_output, real_label)
                d_loss_real.backward()
                d_x =  real_output.mean()

                noise = torch.randn(32)
                noise = noise.cuda()
                fake = generator(graph_, edge_, noise)
                for i in range(len(fake)):
                    fake[i] = list(fake[i].cpu().detach().numpy()[0])[0]
                fake = torch.tensor(fake, dtype=torch.float32)
                fake = fake.unsqueeze(0).cuda()
                fake_output = discriminator(graph_,edge_,fake)
                d_loss_fake = adversarial_criterion(fake_output, fake_label)
                d_loss_fake.backward()
                d_g_z1 = fake_output.mean()

                d_loss = d_loss_real + d_loss_fake
                discriminator_opt.step()

                generator.zero_grad()

                fake_output = discriminator(graph_,edge_,fake)
                g_loss = adversarial_criterion(fake_output, real_label)
                g_loss.backward()
                d_g_z2 = fake_output.mean()
                generator_opt.step()
                
                train_n+=1
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                
                # print('each data generator loss:',g_loss.item())
                # print('---')
                # print('each data disciriminator loss:',d_loss.item())
                # print('----------')

            print()
            print()
            print('g_loss:',g_loss.item()/train_n)
            print('d_loss:',d_loss.item()/train_n)
            print()
            print()

            # # print loss val
            # if epoch % log_step == 0:
            #     y_test = []
            #     X_test = []
            #     y_score = []
            #     with torch.no_grad():
            #         for graph_, edge_, label_ in val_loader:
            #             output = GNN_model(graph_, edge_)
            #             tmp_val_output = (output>probability_threshold).float()
            #             val_acc += (tmp_val_output == label_.view(1,-1,1)).float().sum().item()
                        
            #             y_test += label_.tolist()[0]
            #             for i in output.tolist()[0]:
            #                 y_score += i
            #             X_test += [graph_, edge_]
            #             val_n += output.size(1)
                        
            #     print(f'epoch : {epoch+1}\nloss : {total_loss/len(train_set)}   train_acc : {train_acc/train_n * 100}   val_acc : {val_acc/val_n * 100}')
            #     average_precision_recall = average_precision_score(y_test,y_score)
            #     print(f'average precision-recall score: {average_precision_recall}')
            #     print()

            #     if epoch == epochs-1:
            #         if plot_mAP:
            #             average_precision_recall_plot(y_test, y_score)
                
            # # save model
            # if (epoch+1) % save_step == 0:
            #     test_acc = 0
            #     test_n = 0
            #     with torch.no_grad():
            #         for graph_, edge_, label_ in test_loader:
            #             output = GNN_model(graph_, edge_)
            #             tmp_test_output = (output>probability_threshold).float()
            #             test_acc += (tmp_test_output == label_.view(1,-1,1)).float().sum().item()
            #             test_n += output.size(1)
                        
            #     torch.save(GNN_model.state_dict(),f'./save_model/{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
            #     print(f'save model\ntest_acc : {test_acc/test_n}')
                
    else:
        GNN_model.load_state_dict(torch.load(model_path))
        GNN_model.eval()    
        
        num = int(input(f"# of {len(graph_inputs)} data: "))
        
        graph_, edge_, label_ = graph_inputs[num]
        graph_ = graph_.unsqueeze(0)
        edge_ = edge_.unsqueeze(0)
        label_ = label_.unsqueeze(0)

        print(graph_.size())
        print(edge_.size())
        exit()
        pred = GNN_model(graph_, edge_)
        
        print(label_)
        print(pred)





