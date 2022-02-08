import os, sys, glob
sys.path.append('/home/han811/Desktop/ws/lab/MCD_motion_planner')
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
import shapely.geometry as sg

import torch
import torch.nn as nn

from graph import Graph
from Object import Obstacle


# initialize model weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


# key_configuration generation
def key_configuration_generation():
    key_configuration = []
    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/MCD/data'
    for f in glob.glob(current_path+'/data_*.npy'):
        data_set = np.load(f,allow_pickle=True)
        for data in data_set:
            if data[-1]:
                for node in data[2]['P'][1:-1]:
                    key_configuration.append(list(data[0]['V'][node]))
    key_configuration_path = os.getcwd()
    key_configuration_path += '/key_configuration.npy'
    np.save(key_configuration_path,np.array(key_configuration))


# key configuration load
def key_configuration_load():
    key_configuration_path = os.getcwd()
    key_configuration_path += '/key_configuration.npy'
    return np.load(key_configuration_path,allow_pickle=True)


# graph processing
def graph_processing(key_configurations):
    x = []
    edge = []
    y = []

    current_path = os.getcwd()
    current_path = current_path.split("/")
    tmp_current_path = ''
    for i in current_path[:-1]:
        tmp_current_path += '/'
        tmp_current_path += i
    current_path = tmp_current_path + '/MCD/data'
    for f in glob.glob(current_path+'/data_*.npy'):
        print(f)
        data_set = np.load(f,allow_pickle=True)
        for data in data_set:
            
            tmp_x = []
            tmp_edge = []
            tmp_y = []

            if data[-1]:
                data_ob = data[3].movable_obstacles
                for idx in range(len(data_ob)):
                    tmp_x2 = []
                    tmp_edge2 = [1 for _ in range(len(data_ob))]
                    tmp_edge2[idx] = 0
                    for node in key_configurations:
                        tmp_ob = sg.Point(node[0],node[1])
                        if not data_ob[idx].feasibility_check(tmp_ob):
                            tmp_x2.append(1)
                        else:
                            tmp_x2.append(0)
                    tmp_x.append(list(tmp_x2))
                    tmp_edge.append(list(tmp_edge2))
                
                for d in data[2]['D']:
                    if d!=0:
                        tmp_y.append(1)
                    else:
                        tmp_y.append(0)
            
            if len(tmp_y)!=0:
                x.append(list(tmp_x))
                edge.append(list(tmp_edge))
                y.append(list(tmp_y))
    
    return x, edge, y


# graph generate
def graph_generate(cuda=False, is_key_config=True):
    g = Graph(cuda=cuda)
    if is_key_config:
        key_configuration_generation()
    key_configurations = key_configuration_load()
    
    x, edge, y = graph_processing(key_configurations)
    for idx in range(len(y)):
        g.add_graph(x[idx],edge[idx],y[idx])

    return g, len(key_configurations), y


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
    current_path = tmp_current_path + '/MCD/data'
    for f in glob.glob(current_path+'/data_*.npy'):
        data_set = np.load(f,allow_pickle=True)
        for data in data_set:
            if data[-1]:
                for d in data[2]['D']:
                    if d!=0:
                        num_1+=1
                    else:
                        num_0+=1
    print(f"count 0 : {num_0}")            
    print(f"count 1 : {num_1}")     
    return [num_0, num_1]


# calculate precision recall curve
def average_precision_recall_plot(y_test,y_score):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    for idx in range(len(precision)):
        plt.scatter(recall[idx], precision[idx])
    plt.xlabel = 'recall'
    plt.ylabel = 'precision'
    plt.savefig('precision_recall_curve.png')  

# plot with label
def plot_with_labels(label):
    start_point = plt.Circle((-20, 0), 0.5, color='r')
    goal_point = plt.Circle((20, 0), 0.5, color='b')

    fig, ax = plt.subplots()

    plt.xlim(-30,30)
    plt.ylim(-30,30)

    ax.add_artist(start_point)
    ax.add_artist(goal_point)

    plt.title('Plot validation', fontsize=10)

    for ob_x, ob_y in label[2]:
        obstacle= plt.Circle((ob_x, ob_y), 1.5, color='black', alpha=0.2)
        ax.add_artist(obstacle)
    plt.show()

if __name__=='__main__':
    count_label_01()    
    key_configuration_generation()
    key_configurations = key_configuration_load()
    x, edge, y = graph_processing(key_configurations)
    np.save("./train_data.npy",np.array([x, edge, y]))
    # print(g.label)
    # print(l)


