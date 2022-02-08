'''
Minimum Constraint Displacement planner implementation
Author : han811
email : kimuw42@yonsei.ac.kr

test for importance of key obstacles & displacements
'''

import math
import random
import time
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from Object import Robot, Obstacles, Obstacle
# from Plot_utils import plot_arrow, plot_robot, plot_obs, plot_world
from MCD import MCD

# from my_GNN import model, utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num',default='0',
                        help='0 : not given, 1 : given')
    parser.add_argument('--total_iter',default='0',
                        help='total iteration num')

    args = parser.parse_args()
    data_num = int(args.num)
    total_itertime = int(args.total_iter)
    
    width = 20
    height = 20

    static_rectangle_obs = [[[-5,-10],[5,-10],[5,-5],[-5,-5],[-5,-10]],[[5,10],[5,5],[-5,5],[-5,10],[5,10]]]

    start_position = np.array([-0.5,0.0])
    start_position[0] *= width

    goal_position = np.array([0.5,0.0])
    goal_position[0] *= width
    
    
    ob = Obstacles(num=15, obs_radius=1.8, static_obstacles=static_rectangle_obs, mwidth=width, mheight=height)

    planner = MCD(ob=ob, num=ob.movable_ob_num, width=width, height=height, del_J=2.5, w_l=0.5, w_o=0.1, q_s=start_position, q_g=goal_position, is_GNN=False, visualize=True, record_time=False)
    

    for iter_num in range(total_itertime):
        print("iter_num:",iter_num)
        planner.continuousMCD()
    file_name = f"data_{data_num}.npy"
    planner.save_dataset(file_name=file_name)
    

    
