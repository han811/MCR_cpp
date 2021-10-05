from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from myGNN_preprocessing import train_validation_test_data, return_batch_subgraph
from myGNNs import myGraphAE
from myGNN_utils import get_n_params

from myGNN_config import GAE_config

parser = argparse.ArgumentParser(description='Training my GAE models')

parser.add_argument('--epochs', required=False, default=512, type=int, help='num of epochs')
parser.add_argument('--device', required=False, default='cuda', type=str, help='whether CUDA use or not')
parser.add_argument('--learning_rate', required=False, default=1e-2, type=float, help='learning rate of training')
parser.add_argument('--width', required=False, default=12, type=float, help='width size of map')
parser.add_argument('--height', required=False, default=8, type=float, help='height size of map')

args = parser.parse_args()

epochs = args.epochs
device = args.device
learning_rate = args.learning_rate
width = args.width
height = args.height

if __name__=='__main__':
    train_indexs, validation_indexs, test_indexs = train_validation_test_data((0.7, 0.2, 0.1))
    model = myGraphAE(**GAE_config)
    print(model)
    print(f'model prameter size: {get_n_params(model)}')
    epochs = 100
    device = 'cuda'
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    loss_function = nn.MSELoss()

    model = model.to(device)
    loss_function = loss_function.to(device)
    
    n_obstacles = 25
    n_train_set = len(train_indexs) * n_obstacles
    n_test_set = len(test_indexs) * n_obstacles
    pre_test_avg_mse = -1
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
        avg_mse = 0
        for sub_graphs in return_batch_subgraph(train_indexs, k=3, width=width, height=height):
            for sub_graph in sub_graphs:
                sub_graph.x = torch.tensor(sub_graph.x,dtype=torch.float32)
                sub_graph = sub_graph.to(device)
                optimizer.zero_grad()
                reconstruction_x = model(sub_graph.x,sub_graph.edge_index)
                loss = loss_function(reconstruction_x,sub_graph.x)
                loss.backward()
                optimizer.step()
                avg_mse += loss.item()
        avg_mse /= n_train_set
        print(f'epoch mse: {avg_mse}')
    
        with torch.no_grad():
            test_avg_mse = 0
            for sub_graphs in return_batch_subgraph(test_indexs, k=3, width=width, height=height):
                for sub_graph in sub_graphs:
                    sub_graph.x = torch.tensor(sub_graph.x,dtype=torch.float32)
                    sub_graph = sub_graph.to(device)
                    reconstruction_x = model(sub_graph.x,sub_graph.edge_index)
                    loss = loss_function(reconstruction_x,sub_graph.x)
                    avg_mse += loss.item()
            print(sub_graph.x.view(-1))
            print(reconstruction_x.view(-1))
            test_avg_mse /= n_test_set
            print(f'epoch test mse: {test_avg_mse}')
            if pre_test_avg_mse < test_avg_mse:
                pre_test_avg_mse = test_avg_mse
                torch.save(model.state_dict(), f'./save_model/AE/my_model_AE_{datetime.now().strftime("%Y-%m-%d")}_{datetime.now().strftime("%H-%M")}.pt')



        