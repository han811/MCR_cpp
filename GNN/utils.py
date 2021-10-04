import os, sys, glob
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from graph import Graph

from data_class import MCRdata, GraphSaveClass, MyGraphSaveClass


# initialize model weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

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

if __name__=='__main__':
    count_label_01()