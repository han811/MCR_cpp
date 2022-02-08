'''
Minimum Constraint Displacement planner implementation
Author : han811
email : kimuw42@yonsei.ac.kr
'''

import math
import random
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

import torch
import torch.nn as nn
from torch.serialization import load
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from Object import Obstacles
from MCD import MCD

if __name__ == '__main__':
    width = 20
    height = 20

    static_rectangle_obs = [[[-5,-10],[5,-10],[5,-5],[-5,-5],[-5,-10]],[[5,10],[5,5],[-5,5],[-5,10],[5,10]]]
    # static_rectangle_obs = []

    # start_position = np.random.rand(2) - 0.5
    # start_position[0] *= width
    # start_position[1] *= height

    # goal_position = np.random.rand(2) - 0.5
    # goal_position[0] *= width
    # goal_position[1] *= height

    start_position = np.array([-0.5,0.0])
    start_position[0] *= width

    # goal_position = np.array([-0.3,0.0])
    goal_position = np.array([0.5,0.0])
    goal_position[0] *= width
    
    
    ob = Obstacles(num=10, obs_radius=1.75, static_obstacles=static_rectangle_obs, mwidth=width, mheight=height)
    planner = MCD(ob=ob, num=ob.movable_ob_num, width=width, height=height, del_J=2.5, w_l=0.5, w_o=0.1, q_s=start_position, q_g=goal_position, is_GNN=False, visualize=True)
    
    for iter_num in range(100):
        print("iter_num:",iter_num)
        # for i in range(ob.movable_ob_num):
        #     print(ob.movable_obstacles[i].x,ob.movable_obstacles[i].y)
        planner.continuousMCD()
    
    

    
