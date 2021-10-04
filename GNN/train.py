import os, sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

from model import Node_Update_Function, Edge_to_Node_Aggregation_Function, GNN
from graph import Graph
# from utils import init_weights, graph_generate, count_label_01, average_precision_recall_plot, key_configuration_load
from utils import graph_generate_load, init_weights
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
    print("Data preprocessing start!!")
    # graph_inputs, in_node, labels = graph_generate_load()
    _, in_node, _ = graph_generate_load()
    in_node += 4

    GNN_model = GNN(in_node,100,in_node,3,4,cuda=CUDA)
    GNN_model.apply(init_weights)
    print("------------------------")
    print("Data preprocessing end!!")

    # make a dataset      
    # train_length = int(len(graph_inputs) * 0.7)
    # train_set, val_test_set, train_label, val_test_label = train_test_split(graph_inputs, labels, test_size=0.3)
    
    # val_length = int((len(graph_inputs) - train_length) * 2 / 3)
    # test_length = len(graph_inputs) - train_length - val_length
    # val_set, test_set, val_label, test_label = train_test_split(val_test_set, val_test_label, test_size=0.33)

    train_set_path = os.getcwd()
    train_set_path += '/data/train_set.pickle'
    with open(train_set_path,'rb') as f:
        train_set, train_label = pickle.load(f)

    val_set_path = os.getcwd()
    val_set_path += '/data/validation_set.pickle'
    with open(val_set_path,'rb') as f:
        val_set, val_label = pickle.load(f)

    test_set_path = os.getcwd()
    test_set_path += '/data/test_set.pickle'
    with open(test_set_path,'rb') as f:
        test_set, test_label = pickle.load(f)


    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)
    
    # define loss & optimizer
    optimizer = optim.Adam(GNN_model.parameters(), lr=learning_rate)
    # weights = count_label_01()
    # class_weights = torch.FloatTensor(weights).cuda()
    # loss = nn.BCELoss(weight=(class_weights[0]/(class_weights[0]+class_weights[1])))
    loss = nn.BCELoss()

    if CUDA:
        GNN_model.cuda()
        loss.cuda()
    writer = SummaryWriter()
    
    a,b,c, = train_set[0]
    writer.add_graph(GNN_model, (a.unsqueeze(0).cuda(), b.unsqueeze(0).cuda()))

    # train
    if TRAIN:
        for epoch in range(epochs):
            total_loss = 0
            train_acc = 0
            val_acc = 0
            
            train_n = 0
            val_n = 0

            for graph_, edge_, label_ in train_loader:
                
                optimizer.zero_grad()
                if CUDA:
                    graph_ = graph_.cuda()
                    edge_ = edge_.cuda()
                    label_ = label_.cuda()
                output = GNN_model(graph_,edge_)

                train_loss = loss(output, label_.view(1,-1,1))
                train_loss.backward()
                optimizer.step()
                total_loss += train_loss.item()
                
                tmp_train_output = (output>probability_threshold).float()
                train_acc += (tmp_train_output == label_.view(1,-1,1)).float().sum().item()
                train_n += output.size(1)

            for tag, value in GNN_model.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)
            for name, param in GNN_model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch)
            

            # print loss val
            if epoch % log_step == 0:
                test_total_loss = 0
                y_test = []
                X_test = []
                y_score = []
                with torch.no_grad():
                    for graph_, edge_, label_ in val_loader:
                        if CUDA:
                            graph_ = graph_.cuda()
                            edge_ = edge_.cuda()
                            label_ = label_.cuda()
                        output = GNN_model(graph_, edge_)
                        
                        test_loss = loss(output, label_.view(1,-1,1))
                        test_total_loss += test_loss.item()

                        tmp_val_output = (output>probability_threshold).float()
                        val_acc += (tmp_val_output == label_.view(1,-1,1)).float().sum().item()
                        
                        y_test += label_.tolist()[0]
                        for i in output.tolist()[0]:
                            y_score += i
                        X_test += [graph_, edge_]
                        val_n += output.size(1)
                        
                print(f'epoch : {epoch+1}\nloss : {total_loss/len(train_set)}   train_acc : {train_acc/train_n * 100}   val_acc : {val_acc/val_n * 100}')
                # average_precision_recall = average_precision_score(y_test,y_score)
                # print(f'average precision-recall score: {average_precision_recall}')
                writer.add_scalar("train_epoch_loss", total_loss/len(train_set), epoch)
                writer.add_scalar("train_accuracy", train_acc/train_n * 100, epoch)
                writer.add_scalar("val_epoch_loss", test_total_loss/len(test_set), epoch)
                writer.add_scalar("val_accuracy", val_acc/val_n * 100, epoch)
                print()

                # if epoch == epochs-1:
                #     if plot_mAP:
                #         average_precision_recall_plot(y_test, y_score)
                
            # save model
            if (epoch+1) % save_step == 0:
                test_acc = 0
                test_n = 0
                with torch.no_grad():
                    for graph_, edge_, label_ in test_loader:
                        if CUDA:
                            graph_ = graph_.cuda()
                            edge_ = edge_.cuda()
                            label_ = label_.cuda()
                        output = GNN_model(graph_, edge_)
                        tmp_test_output = (output>probability_threshold).float()
                        test_acc += (tmp_test_output == label_.view(1,-1,1)).float().sum().item()
                        test_n += output.size(1)
                        
                torch.save(GNN_model.state_dict(),f'./save_model/{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
                print(f'save model\ntest_acc : {test_acc/test_n}')
                
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





