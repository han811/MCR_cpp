'''
Minimum Constraint Displacement planner implementation
Author : han811
email : kimuw42@yonsei.ac.kr
'''
import os
import math
import random
import time
import sys
import os
cp = os.getcwd()
sys.path.append(cp)
sys.path.append(os.path.join(cp,'MCD'))
sys.path.append(os.path.join(cp,'my_GNN'))

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from MCD.Object import Obstacles, Obstacle
from MCD.MCD import MCD
from my_GNN.model import GNN
from my_GNN.utils import graph_generate, key_configuration_load
from config import test_config

CUDA = test_config['CUDA']
probability_threshold = test_config['probability_threshold']
same_iter_num = test_config['same_iter_num']
model_path = test_config['model_path']


if __name__ == '__main__':
    width = 20
    height = 20

    static_rectangle_obs = [[[-5,-10],[5,-10],[5,-5],[-5,-5],[-5,-10]],[[5,10],[5,5],[-5,5],[-5,10],[5,10]]]

    start_position = np.array([-0.5,0.0])
    start_position[0] *= width

    goal_position = np.array([0.5,0.0])
    goal_position[0] *= width
    
    with torch.no_grad():
        total_J_list = []
        total_gnn_J_list = []
        sample_num=0
        while True:
            sample_num+=1
            ob = Obstacles(num=10, obs_radius=1.8, static_obstacles=static_rectangle_obs, mwidth=width, mheight=height)
            ob.movable_obstacles = [0,1,2,3,4,5,6,7,8,9]
            ob.movable_obstacles[0] = Obstacle(init_position=(-10.0,0.0), radius=1.2)
            ob.movable_obstacles[1] = Obstacle(init_position=(-5.0,2.5), radius=1.2)
            ob.movable_obstacles[2] = Obstacle(init_position=(-5.0,-2.5), radius=1.2)
            ob.movable_obstacles[3] = Obstacle(init_position=(0.0,1.2), radius=1.2)
            ob.movable_obstacles[4] = Obstacle(init_position=(0.0,3.6), radius=1.2)
            ob.movable_obstacles[5] = Obstacle(init_position=(0.0,-1.2), radius=1.2)
            ob.movable_obstacles[6] = Obstacle(init_position=(0.0,-3.6), radius=1.2)
            ob.movable_obstacles[7] = Obstacle(init_position=(5.0,2.5), radius=1.2)
            ob.movable_obstacles[8] = Obstacle(init_position=(5.0,-2.5), radius=1.2)
            ob.movable_obstacles[9] = Obstacle(init_position=(10.0,0.0), radius=1.2)

            
            J_list = []
            gnn_J_list = []
            for sub_iter_num in range(same_iter_num):
                print("sub_iter_num:",sub_iter_num)
                planner = MCD(ob=ob, num=ob.movable_ob_num, width=width, height=height, del_J=2.5, w_l=0.5, w_o=0.1, q_s=start_position, q_g=goal_position, is_GNN=False, visualize=True)
                J = 0
                for iter_num in range(15):
                    print("MCD_iter_num:",iter_num)
                    J = planner.continuousMCD()
                    J_list.append(J)
                print("no GNN cost value:",J)
                
                key_configuration = np.load("./my_GNN/key_configuration.npy",allow_pickle=True)
                in_node = len(key_configuration)
                GNN_model = GNN(in_node,100,in_node,3,4,cuda=CUDA)
                GNN_model.cuda()
                GNN_model.load_state_dict(torch.load("./my_GNN/save_model/2021-08-04_15-35.pt"))
                GNN_model.eval()
                planner_GNN = MCD(ob=ob, num=ob.movable_ob_num, width=width, height=height, del_J=2.5, w_l=0.5, w_o=0.1, q_s=start_position, q_g=goal_position, is_GNN=True, GNN_model=GNN_model,key_configuration=key_configuration, lambda_value=0.5, visualize=True)
                J_gnn = 0
                for iter_num in range(15):
                    print("GNN_iter_num:",iter_num)
                    J_gnn = planner_GNN.continuousMCD()
                    gnn_J_list.append(J_gnn)
                print("GNN final cost value:",J_gnn)
                
                total_J_list.append(list(J_list))
                total_gnn_J_list.append(list(gnn_J_list))
                
                current_path = os.getcwd()
                if os.path.isfile(os.path.join(current_path,str(sample_num)+"_J_log.npy")):
                    Js = np.load(os.path.join(current_path,str(sample_num)+"_J_log.npy"),allow_pickle=True)
                    Js_gnn = np.load(os.path.join(current_path,str(sample_num)+"_J_gnn_log.npy"),allow_pickle=True)
                    Js = list(Js)
                    Js += total_J_list
                    Js_gnn = list(Js_gnn)
                    Js_gnn += total_gnn_J_list
                    np.save(str(sample_num)+"_J_log.npy",np.array(Js))
                    np.save(str(sample_num)+"_J_gnn_log.npy",np.array(Js_gnn))
                else:
                    np.save(str(sample_num)+"_J_log.npy",np.array(total_J_list))
                    np.save(str(sample_num)+"_J_gnn_log.npy",np.array(total_gnn_J_list))
                
                print("MCD")
                print(total_J_list)
                print("GNN")
                print(total_gnn_J_list)
    

    
