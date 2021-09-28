from datetime import datetime
from typing import *

import torch
import torch.nn as nn
import torch.utils.data as data

import torch.optim as optim
from tensorboardX import SummaryWriter

from torch_geometric.loader import DataLoader

from tg_Preprocessing import load_tg_data

from myGNNs import myGCN, myGraphSAGE, myGAT

# setting parameters
epochs = 128
learning_rate = 1e-3
CUDA = True
log_step = 10
save_step = 400
TRAIN = True
probability_threshold = 0.5
batch_size = 1
in_node = 1199
data_size = (0,3)

if __name__=='__main__':

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

    # clf_model = myGCN(in_channels=in_node, hidden_channels=512, dropout=0.2, activation='elu')
    # clf_model = myGraphSAGE(in_channels=in_node, hidden_channels=512, dropout=0.2, activation='elu')
    clf_model = myGAT(in_channels=in_node, hidden_channels=512, dropout=0.2, activation='elu')
    print(clf_model)

    optimizer = optim.Adam(clf_model.parameters(),lr=learning_rate, betas=(0.5, 0.999), weight_decay=5e-4)
    loss_function = nn.BCELoss()

    if CUDA:
        clf_model.cuda()
        loss_function.cuda()

    # writer = SummaryWriter('GNN_model_test/GraphSAGE')
    writer = SummaryWriter('GNN_model_test/GAT')

    n_train_loader = len(train_loader)
    n_train_set = len(train_set)
    n_obstacles = 25
    n_test_loader = len(test_loader)
    n_test_set = len(test_set)

    if TRAIN:
        pre_accuracy = 0
        current_accuracy = 0
        tmp = train_set[0]
        tmp = tmp.cuda()
        writer.add_graph(clf_model, (tmp.x, tmp.edge_index))
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}')
            avg_loss = 0
            avg_accuracy = 0
            for batch_idx, data in enumerate(train_loader):
                if CUDA:
                    data = data.cuda()
                optimizer.zero_grad()
                # prediction_y = clf_model(data.x,data.edge_index)
                prediction_y, attentions = clf_model(data.x,data.edge_index)
                loss = loss_function(prediction_y,data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                prediction_y_acc = prediction_y>probability_threshold
                avg_accuracy += (prediction_y_acc == data.y.unsqueeze(1)).sum().item()
                if batch_idx % 1000 == 0:
                    print(prediction_y.view(-1))
                    print(data.y.view(-1))
                    # print(attentions[1][1][0].mean())
                    # print(attentions[1][1][5].mean())
                    # print(attentions[1][1][14*24+16].mean())
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
                    # prediction_y = clf_model(data.x,data.edge_index)
                    prediction_y, attentions = clf_model(data.x,data.edge_index)
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
            # if current_accuracy >= pre_accuracy:
            #     torch.save(clf_model.state_dict(), f'./save_model/tg_GNNclf_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
            # with open(f'./weights/weights_{epoch}','wb') as f:
            #     pickle.dump(clf_model.state_dict(),f,pickle.HIGHEST_PROTOCOL)
            torch.save(clf_model.state_dict(), f'./save_model/tg_GNNclf_GAT_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')
