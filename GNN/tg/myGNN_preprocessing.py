import os, glob
from typing import *
import math
import random

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_cluster import knn_graph

from sklearn.model_selection import train_test_split

from tqdm import tqdm

def key_configurations_generation(delta=0.1):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    f_total_num = len(glob.glob(current_path+'/data_npy/graph_node/*'))

    for f_idx in tqdm(range(1,f_total_num+1), desc="key configuration calculation"):
        graph = np.load(current_path+f'/data_npy/graph_node/graph_node{f_idx}.npy')
        traj = np.load(current_path+f'/data_npy/graph_traj/graph_traj{f_idx}.npy')
        for node_idx in traj[1:-1]:
            sig = True
            for key_configuration_i in key_configurations:
                if(math.sqrt((key_configuration_i[0]-graph[node_idx][0])*(key_configuration_i[0]-graph[node_idx][0])+(key_configuration_i[1]-graph[node_idx][1])*(key_configuration_i[1]-graph[node_idx][1]))<delta):
                    sig = False
                    break
            if sig:
                key_configurations.append(list(graph[node_idx]))

    key_configurations_path = os.getcwd()
    key_configurations_path += '/data/key_configurations'
    np.save(key_configurations_path, np.array(key_configurations, dtype=np.float16))

def key_configurations_load():
    key_configurations_path = os.getcwd()
    key_configurations_path += '/data/key_configurations.npy'
    return np.load(key_configurations_path)

def obstacle_graph_processing(width=12.0,height=8.0):
    print('key configurations loading start')
    key_configurations = key_configurations_load()
    print('key configurations loading end')

    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    f_total_num = len(glob.glob(current_path+'/data_npy/graph_node/*'))
    
    for f_idx in tqdm(range(1,f_total_num+1), desc="obstacle graph generation process"):
        circles = np.load(current_path+f'/data_npy/graph_circle/graph_circle{f_idx}.npy')
        ob_label = np.load(current_path+f'/data_npy/graph_label/graph_label{f_idx}.npy')
        graph = np.load(current_path+f'/data_npy/graph_node/graph_node{f_idx}.npy')
        start = graph[0]
        start[0] /= width
        start[1] /= height
        goal = graph[1]
        goal[0] /= width
        goal[1] /= height

        x = []
        y = []
        start_goal = []

        for ob_idx in range(len(circles)):
            tmp_x = []
            radius = circles[ob_idx][-1]
            for key_configuration_i in key_configurations:
                if math.sqrt((key_configuration_i[0]-circles[ob_idx][0])*(key_configuration_i[0]-circles[ob_idx][0])+(key_configuration_i[1]-circles[ob_idx][1])*(key_configuration_i[1]-circles[ob_idx][1]))<radius:
                    tmp_x.append(1)
                else:
                    tmp_x.append(0)
            start_goal += start.tolist()
            start_goal += goal.tolist()
            x.append(tmp_x.copy())
            if (ob_idx+2) in ob_label:
                y.append(1)
            else:
                y.append(0)
        
        obstacle_graph_path = os.getcwd()
        obstacle_graph_path += f'/data/obstacle_graph/obstacle_graph{f_idx}'
        np.save(obstacle_graph_path, np.array(x, dtype=np.int8))
        
        obstacle_label_path = os.getcwd()
        obstacle_label_path += f'/data/obstacle_label/obstacle_label{f_idx}'
        np.save(obstacle_label_path, np.array(y, dtype=np.int8))
        
        graph_start_goal_path = os.getcwd()
        graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{f_idx}'
        np.save(graph_start_goal_path, np.array(start_goal, dtype=np.float16))

def train_validation_test_data(ratio=(0.7, 0.2, 0.1)):
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    f_total_num = len(glob.glob(current_path+'/data_npy/graph_node/*'))
    f_num_list = [i for i in range(1,f_total_num+1)]
    random.shuffle(f_num_list)
    
    train_validation_split = int(f_total_num * 0.7)
    validation_test_split = int(f_total_num * 0.9)

    train_indexs = f_num_list[:train_validation_split]
    validation_indexs = f_num_list[train_validation_split:validation_test_split]
    test_indexs = f_num_list[validation_test_split:]

    return train_indexs, validation_indexs, test_indexs

def knn_subgraph(indexs, k=3, width=12.0, height=8.0):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    obstacle_graph_path = os.getcwd()
    
    key_configurations = key_configurations_load()

    graphs = []
    edge_indexs = []

    for index in indexs:
        tmp_graph = []
        tmp_edge = []

        graph = np.load(current_path+f'/data_npy/graph_node/graph_node{index}.npy')
        obstacle_graph = np.load(obstacle_graph_path+f'/data/obstacle_graph/obstacle_graph{index}.npy')
        
        for obstacle in obstacle_graph:
            key_configuration_mapped = key_configurations[obstacle==1]
            for idx in range(len(key_configuration_mapped)):
                key_configuration_mapped[idx][0] /= width
                key_configuration_mapped[idx][1] /= height
            graph_tensor = torch.Tensor(key_configuration_mapped)
            batch = torch.tensor([0 for _ in range(len(key_configuration_mapped))])
            knn_result = knn_graph(graph_tensor, k=k, batch=batch, loop=False)
            tmp_graph.append(key_configuration_mapped.tolist())
            tmp_edge.append(knn_result.tolist())
        graphs.append(tmp_graph.copy())
        edge_indexs.append(tmp_edge.copy())
    return graphs, edge_indexs

def return_batch_subgraph(indexs, k=3, width=12.0, height=8.0):
    graphs, edge_indexs = knn_subgraph(indexs, k=k, width=width, height=height)
    for graph, edge_index in zip(graphs,edge_indexs):
        sub_graphs = []
        for obstacle_graph, obstacle_edge_index in zip(graph, edge_index):
            tmp_data = Data(x=obstacle_graph.copy(), edge_index=torch.LongTensor(obstacle_edge_index.copy()), y=obstacle_graph.copy())
            sub_graphs.append(tmp_data)
        yield sub_graphs



def draw_obstacle(nodes, edges, width=12.0, height=8.0, number=0):
    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    plt.cla()
    for node in nodes:
        plt.gca().scatter(node[0],node[1],c='red',s=2*0.8*2000.0,alpha=0.1)
    edge_pair = zip(edges[0], edges[1])
    for pair in edge_pair:
        plt.gca().plot([nodes[pair[0]][0],nodes[pair[1]][0]],[nodes[pair[0]][1],nodes[pair[1]][1]],c='black')
    
    plt.savefig(f'./images/obstacle{number}.png')


if __name__=='__main__':
    # key_configurations_generation(0.2)
    # key_configurations = key_configurations_load()
    # obstacle_graph_processing()
    # train_validation_test_data()
    # knn_subgraph([1,2,3])
    for i in return_batch_subgraph([1,3]):
        print(i)
    # print(key_configurations)
