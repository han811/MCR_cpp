import os, sys, glob
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn

from graph import Graph

from data_class import MCRdata, GraphSaveClass, MyGraphSaveClass


# initialize model weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

# key_configuration generation
def key_configuration_generation(delta=0.1):
    key_configuration = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    for f_name in glob.glob(current_path+'/*.pickle'):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            total_num = len(data.graph)
            for idx in range(total_num):
                print(f"current state: {idx/total_num*100}%",end='\r')
                graph = data.graph[idx]
                traj = data.traj[idx]
                for node_idx in traj[1:-1]:
                    sig = True
                    for key_configuration_i in key_configuration:
                        if(math.sqrt((key_configuration_i[0]-graph['V'][node_idx][0])*(key_configuration_i[0]-graph['V'][node_idx][0])+(key_configuration_i[1]-graph['V'][node_idx][1])*(key_configuration_i[1]-graph['V'][node_idx][1]))<delta):
                            sig = False
                            break
                    if sig:
                        key_configuration.append(list(graph['V'][node_idx]))
    key_configuration_path = os.getcwd()
    key_configuration_path += '/key_configuration.pickle'
    with open(key_configuration_path,'wb') as f:
        pickle.dump(key_configuration, f, pickle.HIGHEST_PROTOCOL)

# key configuration load
def key_configuration_load():
    key_configuration_path = os.getcwd()
    key_configuration_path += '/key_configuration.pickle'
    with open(key_configuration_path,'rb') as f:
        return pickle.load(f)

# count label 0 & 1    
def count_label_01():
    
    num_0 = 0
    num_1 = 0
    
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    for f_name in glob.glob(current_path+'/*.pickle'):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            for idx in range(len(data.ob_label)):
                ob_label = data.ob_label[idx]
                circle = data.circle[idx]
                num_1+=len(ob_label)
                num_0+=len(circle)-len(ob_label)
    print(f"count 0 : {num_0}")            
    print(f"count 1 : {num_1}")     
    print(f"ratio 0 & 1: {num_0/(num_0+num_1)} & {num_1/(num_0+num_1)}")
    return [num_0, num_1]

# plot key configurations
def plot_key_configurations(width=12.0,height=8.0,is_key_configuration=False):
    if not is_key_configuration:
        key_configuration_generation()
    key_configurations = key_configuration_load()

    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    for key_configuration in key_configurations:
        plt.gca().scatter(key_configuration[0],key_configuration[1],c='blue',s=0.1,alpha=1.0)
    plt.savefig(f'./images/key_configurations.png')

# graph processing
def graph_processing(is_key_configuration=False):
    x = []
    edge = []
    y = []

    print('key configuration loading start')
    if not is_key_configuration:
        key_configuration_generation()
    key_configuration = key_configuration_load()
    print('key configuration loading end')

    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    for f_name in glob.glob(current_path+'/*.pickle'):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            total_num = len(data.graph)
            for idx in range(total_num):
                print(f'graph processing state: {idx/total_num*100}%',end='\r')
                circles = data.circle[idx]
                radius = data.radius[idx]
                ob_label = data.ob_label[idx]
                graph = data.graph[idx]
                start = graph['V'][0]
                goal = graph['V'][1]
                tmp_x = []
                tmp_edge = []
                tmp_y = []

                for ob_idx,circle in circles.items():
                    tmp_x2 = []
                    tmp_edge2 = [1 for _ in range(len(circles))]
                    tmp_edge2[ob_idx] = 0
                    for key_configuration_i in key_configuration:
                        if math.sqrt((key_configuration_i[0]-circle[0])*(key_configuration_i[0]-circle[0])+(key_configuration_i[1]-circle[1])*(key_configuration_i[1]-circle[1]))<radius:
                            tmp_x2.append(1)
                        else:
                            tmp_x2.append(0)
                    tmp_x2 += start
                    tmp_x2 += goal
                    tmp_x.append(tmp_x2.copy())
                    tmp_edge.append(tmp_edge2.copy())
                    if (ob_idx+2) in ob_label:
                        tmp_y.append(1)
                    else:
                        tmp_y.append(0)
                x.append(tmp_x.copy())
                edge.append(tmp_edge.copy())
                y.append(tmp_y.copy())
    train_graph = GraphSaveClass(x,edge,y)
    key_configuration_path = os.getcwd()
    key_configuration_path += '/train_graph.pickle'
    with open(key_configuration_path,'wb') as f:
        pickle.dump(train_graph, f, pickle.HIGHEST_PROTOCOL)
    return x, edge, y

def graph_processing_load():
    key_configuration_path = os.getcwd()
    key_configuration_path += '/train_graph.pickle'
    with open(key_configuration_path,'rb') as f:
        return pickle.load(f)

# graph generate
def graph_generate(is_graph=False, is_key_configuration=False):
    g = Graph()
    if (not is_graph) or (not is_key_configuration):
        graph_processing(is_key_configuration=is_key_configuration)

    train_graph = graph_processing_load()
    key_configurations = key_configuration_load()
    
    x = train_graph.x
    edge = train_graph.edge
    y = train_graph.y

    for idx in range(len(y)):
        g.add_graph(x[idx],edge[idx],y[idx])

    my_data = MyGraphSaveClass(g,len(key_configurations),y)
    my_data_path = os.getcwd()
    my_data_path += '/my_train_graph.pickle'
    with open(my_data_path,'wb') as f:
        pickle.dump(my_data, f, pickle.HIGHEST_PROTOCOL)
    return g, len(key_configurations), y

# graph generate load
def graph_generate_load():
    my_data_path = os.getcwd()
    my_data_path += '/my_train_graph.pickle'
    with open(my_data_path,'rb') as f:
        my_data = pickle.load(f)
    return my_data.g, my_data.key_configuration_size, my_data.y

# plot obstacle graph by key configurations
def plot_obstacle_graph(obstacles, labels, key_configurations, width=12.0, height=8.0, name=None):
    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    for obstacle_idx, obstacle in enumerate(obstacles):
        for idx, v in enumerate(obstacle[:-4]):
            if v:
                key_configuration = key_configurations[idx]
                label_sig=labels[obstacle_idx]
                if label_sig:
                    plt.gca().scatter(key_configuration[0],key_configuration[1],c='orange',s=0.25,alpha=1.0)
                else:
                    plt.gca().scatter(key_configuration[0],key_configuration[1],c='red',s=0.25,alpha=1.0)
    start_point = obstacles[0][-4:-2]
    plt.gca().scatter(start_point[0],start_point[1],c='green',s=0.35,alpha=1.0)
    goal_point = obstacles[0][-2:]
    plt.gca().scatter(goal_point[0],goal_point[1],c='blue',s=0.35,alpha=1.0)
    if name:
        plt.savefig(f'./images/obstacle_graph_{name}.png')
    else:
        plt.savefig(f'./images/obstacle_graph.png')

def plot_obstacle_graph_all(graphs, graph_labels, key_configurations, width=12.0, height=8.0):
    for graph_idx, graph in enumerate(graphs):
        plot_obstacle_graph(graph,graph_labels[graph_idx],key_configurations,name=graph_idx+1)





if __name__=='__main__':
    # count_label_01()    
    # key_configuration_generation()
    # key_configurations = key_configuration_load()
    # plot_key_configurations(is_key_configuration=True)
    # x, edge, y = graph_processing(is_key_configuration=True)
    print(f'loading key_configurations start')
    key_configurations = key_configuration_load()
    print(f'loading key_configurations end')
    # graph_generate(is_graph=True,is_key_configuration=True)
    # graph_generate(True,True)
    # plot_obstacle_graph_all(train_data.x,train_data.y,key_configurations)
    # graph, in_node, y = graph_generate(is_graph=True,is_key_configuration=True)
    # print(x[0])
    # x, edge, y = graph_generate_load()
    # for i in y[:10]:
    #     print(i)
    # _, _, y = graph_generate(is_graph=True,is_key_configuration=True)


