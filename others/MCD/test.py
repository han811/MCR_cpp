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
    parser.add_argument('--given',default='0',
                        help='0 : not given, 1 : given')

    args = parser.parse_args()
    if args.given=='0':
        is_sample_given = False
    else:
        is_sample_given = True
    
    width = 20
    height = 20

    static_rectangle_obs = [[[-5,-10],[5,-10],[5,-5],[-5,-5],[-5,-10]],[[5,10],[5,5],[-5,5],[-5,10],[5,10]]]

    start_position = np.array([-0.5,0.0])
    start_position[0] *= width

    goal_position = np.array([0.5,0.0])
    goal_position[0] *= width
    
    
    ob = Obstacles(num=10, obs_radius=1.2, static_obstacles=static_rectangle_obs, mwidth=width, mheight=height)
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

    planner = MCD(ob=ob, num=ob.movable_ob_num, width=width, height=height, del_J=2.5, w_l=0.5, w_o=0.1, q_s=start_position, q_g=goal_position, is_GNN=False, visualize=False, record_time=False)
    
    if is_sample_given:
        planner.D[0].append([[0.0,1.3],1.3])
        planner.D[0].append([[0.0,-1.31],1.31])
        
        planner.D[3].append([[0.0,1.2],1.2])
        planner.D[3].append([[0.0,2.4],2.4])
        
        planner.D[5].append([[0.0,-1.2],1.2])
        planner.D[5].append([[0.0,-2.4],2.4])
        
        planner.D[9].append([[0.0,1.3],1.3])
        planner.D[9].append([[0.0,-1.31],1.31])
        
        total_itertime = 10
    else:
        total_itertime = 14
        
    
    for iter_num in range(total_itertime):
        print("iter_num:",iter_num)
        planner.continuousMCD()
    if is_sample_given:
        planner.save_dataset(file_name="given.npy")
    else:
        planner.save_dataset(file_name="not_given.npy")
    

    
