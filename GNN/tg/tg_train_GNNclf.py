from datetime import datetime
import pickle
from typing import *

import torch
import torch.nn as nn
import torch.utils.data as data

import torch.optim as optim
from tensorboardX import SummaryWriter

from torch_geometric.loader import DataLoader

from tg_utils import get_n_params, count_label_01_dataset, init_weights, weighted_binary_cross_entropy, FocalLoss
from tg_config import tg_clf_config
from tg_Preprocessing import load_tg_data

from tg_GNNclf import myGCN, GATclf, myGraphSAGE

# setting parameters
epochs = tg_clf_config['epochs']
learning_rate = tg_clf_config['learning_rate']
CUDA = tg_clf_config['CUDA']
log_step = tg_clf_config['log_step']
save_step = tg_clf_config['save_step']
TRAIN = tg_clf_config['TRAIN']
plot_mAP = tg_clf_config['plot_mAP']
probability_threshold = tg_clf_config['probability_threshold']
model_path = tg_clf_config['model_path']
batch_size = tg_clf_config['batch_size']
n_encoding_feature = tg_clf_config['n_encoding_feature']
in_node = tg_clf_config['in_node']
data_size = tg_clf_config['tg_clf_config']


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

    clf_model = myGraphSAGE(in_channels=in_node, enc_channels=1024, hidden_channels=256, num_layers=5, out_channels=512, dropout=0.4, act=nn.ELU(), clf_num_layers=5)
    # clf_model.apply(init_weights)
    print(clf_model)
    # clf_model = GATclf(
    #     in_node, n_encoding_feature,
    #     # n_mid_features = (64, 128, 64),
    #     n_mid_features = (8,),
    #     encoder_hidden_layers = (8,),
    #     node_hidden_layers = ((8,),
    #                           (8,),
    #                           (8,)),
    #     output_hidden_layers = (8,),
    #     message_passing_steps = 1, activation = 'elu')

    # clf_model = GATclf(
    #     in_node, n_encoding_feature,
    #     # n_mid_features = (64, 128, 64),
    #     n_mid_features = (128,),
    #     encoder_hidden_layers = (128, 256, 128),
    #     node_hidden_layers = ((128, 256, 128),
    #                           (128, 256, 128),
    #                           (128, 256, 128)),
    #     output_hidden_layers = (128, 256, 128),
    #     message_passing_steps = 1, activation = 'elu')
    # clf_model.apply(init_weights)

    optimizer = optim.Adam(clf_model.parameters(),lr=learning_rate, betas=(0.5, 0.999), weight_decay=5e-4)
    loss_function = nn.BCELoss()
    # loss_function = weighted_binary_cross_entropy
    # loss_function = FocalLoss(gamma=4,alpha=1e0)

    if CUDA:
        clf_model.cuda()
        loss_function.cuda()

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
        writer.add_graph(clf_model, (tmp.x, tmp.edge_index))
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}')
            avg_loss = 0
            avg_accuracy = 0
            for batch_idx, data in enumerate(train_loader):
                # data = train_set[0]
                if CUDA:
                    data = data.cuda()
                optimizer.zero_grad()
                prediction_y = clf_model(data.x,data.edge_index)
                loss = loss_function(prediction_y,data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                prediction_y_acc = prediction_y>probability_threshold
                avg_accuracy += (prediction_y_acc == data.y.unsqueeze(1)).sum().item()
                if batch_idx % 1000 == 0:
                    print(prediction_y.view(-1))
                    print(data.y.view(-1))
                print(f'each batch: {loss.item()}',end='\r')
            avg_loss /= n_train_loader
            avg_accuracy /= (n_train_set*n_obstacles)
            print(f'epoch loss: {avg_loss}')
            print(f'epoch accuracy: {avg_accuracy*100}%')
            writer.add_scalar("tg_epoch_loss", avg_loss, epoch)
            writer.add_scalar("tg_epoch_accuracy", avg_accuracy, epoch)

            for tag, value in clf_model.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)
            
            # test part
            test_avg_loss = 0
            test_avg_accuracy = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    if CUDA:
                        data = data.cuda()
                    prediction_y = clf_model(data.x,data.edge_index)
                    loss = loss_function(prediction_y,data.y.unsqueeze(1))
                    test_avg_loss += loss.item()
                    prediction_y_acc = prediction_y>probability_threshold
                    test_avg_accuracy += (prediction_y_acc == data.y.unsqueeze(1)).sum().item()
                    print(f'test each batch: {loss.item()}',end='\r')
                test_avg_loss /= n_test_loader
                test_avg_accuracy /= (n_test_set*n_obstacles)
                print(f'test epoch loss: {test_avg_loss}')
                print(f'test epoch accuracy: {test_avg_accuracy*100}%')
                writer.add_scalar("test_epoch_loss", test_avg_loss, epoch)
                writer.add_scalar("test_epoch_accuracy", test_avg_accuracy, epoch)
            for name, param in clf_model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch)
            if epoch==0:
                current_accuracy = test_avg_accuracy
            else:
                pre_accuracy = current_accuracy
                current_accuracy = test_avg_accuracy
            if current_accuracy >= pre_accuracy:
                torch.save(clf_model.state_dict(), f'./save_model/tg_GNNclf_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
            with open(f'./weights/weights_{epoch}','wb') as f:
                pickle.dump(clf_model.state_dict(),f,pickle.HIGHEST_PROTOCOL)