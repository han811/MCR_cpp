import os, glob
import math
import pickle

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from graph import Graph

from data_class import GraphSaveClass, MyGraphSaveClass

# key_configuration generate & load and plot
def key_configurations_generation(delta=0.1):
    key_configurations = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    f_total_num = len(glob.glob(current_path+'/*.pickle'))
    for f_idx, f_name in enumerate(glob.glob(current_path+'/*.pickle')):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            total_num = len(data.graph)
            for idx in range(total_num):
                print(f"{f_idx}/{f_total_num} - {f_name} : key_configurations_generation current state: {idx/total_num*100}%",end='\r')
                graph = data.graph[idx]
                traj = data.traj[idx]
                for node_idx in traj[1:-1]:
                    sig = True
                    for key_configuration_i in key_configurations:
                        if(math.sqrt((key_configuration_i[0]-graph['V'][node_idx][0])*(key_configuration_i[0]-graph['V'][node_idx][0])+(key_configuration_i[1]-graph['V'][node_idx][1])*(key_configuration_i[1]-graph['V'][node_idx][1]))<delta):
                            sig = False
                            break
                    if sig:
                        key_configurations.append(list(graph['V'][node_idx]))
    key_configurations_path = os.getcwd()
    key_configurations_path += '/data/key_configurations.pickle'
    with open(key_configurations_path,'wb') as f:
        pickle.dump(key_configurations, f, pickle.HIGHEST_PROTOCOL)

def key_configurations_load():
    key_configurations_path = os.getcwd()
    key_configurations_path += '/data/key_configurations.pickle'
    with open(key_configurations_path,'rb') as f:
        return pickle.load(f)

def plot_key_configurations(width=12.0,height=8.0,is_key_configuration=False):
    if not is_key_configuration:
        key_configurations_generation()
    key_configurations = key_configurations_load()

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
    plt.savefig(f'./images/key_configurations.png')


# key configuration obstacle graph generate & load
def obstacle_graph_processing(width=12.0,height=8.0,is_key_configuration=False):
    x = []
    edge = []
    y = []

    print('key configurations loading start')
    if not is_key_configuration:
        key_configurations_generation()
    key_configurations = key_configurations_load()
    print('key configurations loading end')

    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/data/CollectedData'
    f_total_num = len(glob.glob(current_path+'/*.pickle'))
    for f_idx, f_name in enumerate(glob.glob(current_path+'/*.pickle')):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            total_num = len(data.graph)
            for idx in range(total_num):
                print(f'{f_idx}/{f_total_num} - {f_name} obstacle graph processing state: {idx/total_num*100}%',end='\r')
                circles = data.circle[idx]
                radius = data.radius[idx]
                ob_label = data.ob_label[idx]
                graph = data.graph[idx]
                start = graph['V'][0]
                start[0] /= width
                start[1] /= height
                goal = graph['V'][1]
                goal[0] /= width
                goal[1] /= height

                tmp_x = []
                tmp_edge = []
                tmp_y = []

                for ob_idx,circle in circles.items():
                    tmp_x2 = []
                    tmp_edge2 = [1 for _ in range(len(circles))]
                    tmp_edge2[ob_idx] = 0
                    for key_configuration_i in key_configurations:
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
    key_configuration_path += '/data/obstacle_graph.pickle'
    with open(key_configuration_path,'wb') as f:
        pickle.dump(train_graph, f, pickle.HIGHEST_PROTOCOL)
    return x, edge, y

def obstacle_graph_processing_load():
    key_configuration_path = os.getcwd()
    key_configuration_path += '/data/obstacle_graph.pickle'
    with open(key_configuration_path,'rb') as f:
        return pickle.load(f)


# dataset graph generate & load
def dataset_graph_generate(is_graph=False, is_key_configuration=False):
    if (not is_graph) or (not is_key_configuration):
        obstacle_graph_processing(is_key_configuration=is_key_configuration)

    obstacle_graph = obstacle_graph_processing_load()
    
    x = obstacle_graph.x
    edge = obstacle_graph.edge
    y = obstacle_graph.y

    total_num = len(y)
    tmp_g = Graph()
    for num in range(int(len(y)/1000)):
        tmp_g.empty()
        for idx in range(1000*num,min(1000*(num+1),total_num)):
            print(f'dataset graph generate state: {idx/total_num*100}%',end='\r')
            tmp_g.add_graph(x[idx],edge[idx],y[idx])
    my_data_path = os.getcwd()
    my_data_path += f'/data/dataset_graph{num}.pickle'
    with open(my_data_path,'wb') as f:
        pickle.dump(tmp_g, f, pickle.HIGHEST_PROTOCOL)

def dataset_graph_generate_load(num=0):
    my_data_path = os.getcwd()
    my_data_path += f'/data/dataset_graph{num}.pickle'
    with open(my_data_path,'rb') as f:
        my_data = pickle.load(f)
    return my_data

# train test validation data split
def train_validation_test_data_split(ratio=(0.7, 0.2, 0.1),num=0):
    graph_inputs, _, _ = dataset_graph_generate_load(num)
    train_set, val_test_set, train_label, val_test_label = train_test_split(graph_inputs, graph_inputs.labels, test_size=(ratio[1]+ratio[2]))
    val_set, test_set, val_label, test_label = train_test_split(val_test_set, val_test_label, test_size=(ratio[2]/(ratio[1]+ratio[2])))

    my_data_path = os.getcwd()
    my_data_path += f'/data/train_set{num}.pickle'
    with open(my_data_path,'wb') as f:
        pickle.dump((train_set, train_label), f, pickle.HIGHEST_PROTOCOL)
    
    my_data_path = os.getcwd()
    my_data_path += f'/data/validation_set{num}.pickle'
    with open(my_data_path,'wb') as f:
        pickle.dump((val_set, val_label), f, pickle.HIGHEST_PROTOCOL)
    
    my_data_path = os.getcwd()
    my_data_path += f'/data/test_set{num}.pickle'
    with open(my_data_path,'wb') as f:
        pickle.dump((test_set, test_label), f, pickle.HIGHEST_PROTOCOL)


# plot obstacle graph by key configurations
def plot_obstacle_graph(obstacles, labels, key_configurations, width=12.0, height=8.0, name=None):
    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, width))
    ax.set_ylim((0.0, height))

    total_obstacle_num = len(obstacles)
    for obstacle_idx, obstacle in enumerate(obstacles):
        print(f'obstacle plot state: {obstacle_idx/total_obstacle_num*100}%',end='\r')
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

def n_plot_obstacle_graph(graphs, graph_labels, key_configurations, nums, width=12.0, height=8.0):
    for graph_idx, graph in enumerate(graphs[:nums]):
        plot_obstacle_graph(graph,graph_labels[graph_idx],key_configurations,name=graph_idx+1)


# whole data preprocessing steps
def whole_process(nums):
    print(f'key configuration generation start!!\n')
    key_configurations_generation(delta=0.05)
    print(f'key configuration generation end!!\n')
    print()

    print(f'key configuration load start!!\n')
    key_configurations = key_configurations_load()
    print(f'key configuration load end!!\n')
    print()

    print(f'key configuration plot start!!\n')
    plot_key_configurations(is_key_configuration=True)
    print(f'key configuration plot end!!\n')
    print()

    print(f'obstacle graph processing start!!\n')
    x, edge, y = obstacle_graph_processing(is_key_configuration=True)
    print(f'obstacle graph processing end!!\n')
    print()

    print(f'dataset graph generate start!!\n')
    dataset_graph_generate(is_graph=True, is_key_configuration=True)
    print(f'dataset graph generate end!!\n')
    print()

    print(f'train validation test dataset split start!!\n')
    train_validation_test_data_split(ratio=(0.7, 0.2, 0.1),num=0)
    print(f'train validation test dataset split end!!\n')
    print()



if __name__=='__main__':
    # print(f'key configuration generation start!!\n')
    # key_configurations_generation(delta=0.05)
    # print(f'key configuration generation end!!\n')
    # print()

    print(f'obstacle graph processing start!!\n')
    x, edge, y = obstacle_graph_processing(is_key_configuration=True)
    print(f'obstacle graph processing end!!\n')
    print()

    print(f'dataset graph generate start!!\n')
    dataset_graph_generate(is_graph=True, is_key_configuration=True)
    print(f'dataset graph generate end!!\n')
    print()

    for i in range(6):
        print(f'train validation test dataset split start!!\n')
        train_validation_test_data_split(ratio=(0.7, 0.2, 0.1),num=i)
        print(f'train validation test dataset split end!!\n')
        print()