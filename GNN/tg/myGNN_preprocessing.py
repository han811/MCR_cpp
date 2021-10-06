import os, glob
from typing import *
import math
import random
from unicodedata import decimal

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph

from tqdm import tqdm


'''
    key configuration part
'''
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
    key_configurations_path += f'/data/key_configurations{delta}'
    np.save(key_configurations_path, np.array(key_configurations, dtype=np.float16))

def key_configurations_load(delta=0.1):
    key_configurations_path = os.getcwd()
    key_configurations_path += f'/data/key_configurations{delta}.npy'
    return np.load(key_configurations_path)

def plot_key_configurations(delta=0.1,width=12.0,height=8.0,is_key_configuration=False):
    if not is_key_configuration:
        key_configurations_generation(delta=delta)
    key_configurations = key_configurations_load(delta=delta)

    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    total_num = len(key_configurations)
    for idx, key_configuration in enumerate(key_configurations):
        plt.gca().scatter(key_configuration[0],key_configuration[1],c='blue',s=0.1,alpha=1.0)
        print(f"key_configurations_plotting current state: {idx/total_num*100}%",end='\r')
    plt.savefig(f'./images/key_configurations{delta}.png')


'''
    one-hot obstacle graph processing
'''
def obstacle_graph_processing(width=12.0,height=8.0,delta=0.1):
    print('key configurations loading start')
    key_configurations = key_configurations_load(delta=delta)
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
        obstacle_graph_path += f'/data/obstacle_graph/obstacle_graph{f_idx}_{delta}'
        np.save(obstacle_graph_path, np.array(x, dtype=np.int8))
        
        obstacle_label_path = os.getcwd()
        obstacle_label_path += f'/data/obstacle_label/obstacle_label{f_idx}_{delta}'
        np.save(obstacle_label_path, np.array(y, dtype=np.int8))
        
        graph_start_goal_path = os.getcwd()
        graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{f_idx}_{delta}'
        np.save(graph_start_goal_path, np.array(start_goal, dtype=np.float16))

def plot_obstacle_graph(index, width=12.0, height=8.0, delta=0.1, name=None):
    key_configurations = key_configurations_load(delta=delta)

    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    obstacle_graph_path = os.getcwd()
    obstacle_graph_path += f'/data/obstacle_graph/obstacle_graph{index}_{delta}.npy'
    obstacle_graph = np.load(obstacle_graph_path)

    obstacle_label_path = os.getcwd()
    obstacle_label_path += f'/data/obstacle_label/obstacle_label{index}_{delta}.npy'
    obstacle_label = np.load(obstacle_label_path)

    graph_start_goal_path = os.getcwd()
    graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{index}_{delta}.npy'
    graph_start_goal = np.load(graph_start_goal_path)

    for obstacle_idx, obstacle in tqdm(enumerate(obstacle_graph), desc='obstacle plot state'):
        for idx, v in enumerate(obstacle):
            if v:
                key_configuration = key_configurations[idx]
                label_sig=obstacle_label[obstacle_idx]
                if label_sig:
                    plt.gca().scatter(key_configuration[0],key_configuration[1],c='orange',s=0.25,alpha=1.0)
                else:
                    plt.gca().scatter(key_configuration[0],key_configuration[1],c='red',s=0.25,alpha=1.0)
    start_point = graph_start_goal[0:2]
    plt.gca().scatter(start_point[0],start_point[1],c='green',s=0.35,alpha=1.0)
    goal_point = graph_start_goal[2:4]
    plt.gca().scatter(goal_point[0],goal_point[1],c='blue',s=0.35,alpha=1.0)
    if name:
        plt.savefig(f'./images/obstacle_graph_{name}.png')
    else:
        plt.savefig(f'./images/obstacle_graph.png')


'''
    train validation test index set split
'''
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
    
    train_validation_split = int(f_total_num * ratio[0])
    validation_test_split = int(f_total_num * (ratio[0] + ratio[1]))

    train_indexs = f_num_list[:train_validation_split]
    validation_indexs = f_num_list[train_validation_split:validation_test_split]
    test_indexs = f_num_list[validation_test_split:]

    return train_indexs, validation_indexs, test_indexs


'''
    make sub_graph for autoencoding
'''
def knn_subgraph(indexs, k=3, width=12.0, height=8.0, delta=0.1):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    obstacle_graph_path = os.getcwd()
    
    key_configurations = key_configurations_load(delta=delta)

    graphs = []
    edge_indexs = []

    for index in indexs:
        tmp_graph = []
        tmp_edge = []

        graph = np.load(current_path+f'/data_npy/graph_node/graph_node{index}_{delta}.npy')
        obstacle_graph = np.load(obstacle_graph_path+f'/data/obstacle_graph/obstacle_graph{index}_{delta}.npy')
        
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


'''
    return all indicator and obstacle configuration coordinates concatenated feature vector
'''
def all_indicator_coordinates_graph(indexs, width=12.0, height=8.0, delta=0.1):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    obstacle_graph_path = os.getcwd()
    
    key_configurations = key_configurations_load(delta=delta)
    graphs = []
    for index in tqdm(indexs, desc='make graph'):
        graph_start_goal_path = os.getcwd()
        graph_start_goal_path += f'/data/graph_start_goal_path/graph_start_goal_path{index}_{delta}.npy'
        start_goal = np.load(graph_start_goal_path)[0:4].tolist()

        tmp_graph = []

        obstacle_graph = np.load(obstacle_graph_path+f'/data/obstacle_graph/obstacle_graph{index}_{delta}.npy')
        
        for obstacle in obstacle_graph:
            obstacle_graph_with_configuration = []
            for key_idx,key_config in enumerate(key_configurations):
                if obstacle[key_idx]:
                    obstacle_graph_with_configuration.append(1)
                else:
                    obstacle_graph_with_configuration.append(0)    
                obstacle_graph_with_configuration.append(key_config[0]/width)
                obstacle_graph_with_configuration.append(key_config[1]/height)
            obstacle_graph_with_configuration += start_goal
            tmp_graph.append(obstacle_graph_with_configuration.copy())

        tmp_y = np.load(os.getcwd()+f'/data/obstacle_label/obstacle_label{index}_{delta}.npy').tolist()
        n_obstacle = len(tmp_graph)
        tmp_edge = []
        tmp_edge_element1 = []
        tmp_edge_element2 = []
        for i in range(n_obstacle):
            for j in range(n_obstacle):
                if i!=j:
                    tmp_edge_element1 += [j]
            tmp_edge_element2 += [i for _ in range(n_obstacle-1)]
        tmp_edge = [tmp_edge_element1, tmp_edge_element2]
        graphs.append(Data(x=torch.tensor(tmp_graph,dtype=torch.float32), edge_index=torch.LongTensor(tmp_edge), y=torch.tensor(tmp_y,dtype=torch.float32)))
    return graphs


'''
    return indicator and obstacle configuration coordinates concatenated feature vector
'''
def indicator_coordinates_graph(indexs, width=12.0, height=8.0, delta=0.1):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    
    for i in current_path[:-2]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    obstacle_graph_path = os.getcwd()
    
    key_configurations = key_configurations_load(delta=delta)

    for index in indexs:
        tmp_graph = []

        obstacle_graph = np.load(obstacle_graph_path+f'/data/obstacle_graph/obstacle_graph{index}_{delta}.npy')
        
        for obstacle in obstacle_graph:
            obstacle_graph_with_configuration = []
            for key_idx,key_config in enumerate(key_configurations):
                if obstacle[key_idx]:
                    obstacle_graph_with_configuration.append(1)
                else:
                    obstacle_graph_with_configuration.append(0)    
                obstacle_graph_with_configuration.append(key_config[0]/width)
                obstacle_graph_with_configuration.append(key_config[1]/height)
            tmp_graph.append(obstacle_graph_with_configuration.copy())
        yield tmp_graph

def draw_indicator_coordinates_graph_for_check(indexs, width=12.0, height=8.0, delta=0.1):
    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    for graphs_idx, graphs in enumerate(indicator_coordinates_graph(indexs, width=12.0, height=8.0, delta=delta)):
        each_obs_feature_length = len(graphs[0])
        for graph in tqdm(graphs, desc='obstacle plot state'):
            for idx in range(int(each_obs_feature_length/3)):
                if graph[idx*3]==1:
                    x = graph[idx*3+1]
                    y = graph[idx*3+2]
                    plt.gca().scatter(x*width,y*height,c='red',s=0.25,alpha=1.0)
        plt.savefig(f'./images/obstacle_graph{indexs[graphs_idx]}_{delta}.png')




if __name__=='__main__':
    # plot_key_configurations(delta=0.2,width=12.0,height=8.0)
    # key_configurations_generation(0.2)
    # key_configurations = key_configurations_load()
    # obstacle_graph_processing(width=12.0,height=8.0,delta=0.1)
    # plot_obstacle_graph(100,width=12.0, height=8.0, delta=0.1)
    # train, val, test = train_validation_test_data()
    draw_indicator_coordinates_graph_for_check([100], width=12.0, height=8.0, delta=0.1)
    # knn_subgraph([1,2,3])
    # for i in return_batch_subgraph([1,3]):
    #     print(i)
    # print(key_configurations)
