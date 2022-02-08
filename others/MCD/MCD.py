import os
import math
import random
from collections import deque
import heapq
from itertools import product
import time
import pickle

import shapely.geometry as sg
import shapely.ops as so

import matplotlib.pyplot as plt
import numpy as np

import torch

from pq import PriorityQueue
# from .pq import PriorityQueue

from Object import Obstacle
from Utils import record_trajectory, cal_J, plot_world, plot_start_with_sol, plot_obs_with_sol, plot_start, plot_sol
from ExpandRoadMap import sample, closest, extendToward, neighbors, discreteMCD_back_checking
from SampleDisplacement import unblocking, refining
from LocalOptimize import optimize_path_only, optimize_displacements
# from Plot_utils import plot_world

# from my_GNN import utils

class MCD:
    def __init__(self,
                 ob,
                 num=100,
                 width=30.0, height=30.0,
                 del_J=0.002, n_e=20, n_o=100,
                 goal_threshold = 3.5,
                 w_l=0.01,w_o=0.1,
                 q_s=(0,0), q_g=(10,10),
                 debug=False,
                 dimension=2,
                 delta=5,
                 is_GNN=False,
                 GNN_model=None,
                 key_configuration=None,
                 lambda_value=0.5,
                 GNN_threshold=0.5,
                 visualize=True,
                 record_time=False,
                 record=False):
        
        # graph
        vertices = dict()
        vertices[0] = q_s.tolist()
        vertices[1] = q_g.tolist()
        
        edge = dict()
        edge[0] = []
        edge[1] = []
        
        self.graph = dict()
        self.graph['V'] = vertices
        self.graph['E'] = edge
        
        self.ob = ob
        self.q_s = q_s
        self.q_g = q_g
        
        # displacement
        self.dimension = dimension
        self.D = [list([[[0.0,0.0],0.0]]) for _ in range(num)]
            
        # trajectory
        self.traj = dict()
        self.traj['P'] = [0,1]
        self.traj['D'] = [[0.0 for _ in range(dimension)] for _ in range(num)]
        self.J = 0 # set to 0 for planning
        
        # parameter setting
        self.del_J = del_J
        self.n_e = n_e
        self.n_o = n_o
        self.w_l = w_l
        self.w_o = w_o
        self.delta = delta
        self.goal_threshold = goal_threshold
        
        # solution presence or absence
        self.is_sol = False
        self.is_new_sol = False
        
        # map
        self.width = width
        self.height = height

        self.debug = debug
        
        # displacement sampling considerations
        self.unblocking_obstacles_set = []
        self.refining_obstacles_count = [0 for _ in range(self.ob.ob_num)]
        
        self.data_set = []
        
        self.is_GNN = is_GNN
        self.GNN_model = GNN_model
        self.key_configuration = key_configuration
        self.lambda_value = lambda_value
        self.GNN_threshold = GNN_threshold
        
        if record_time:
            self.record_time = record_time
            self.start_time = time.time()
            self.first_sol_time = None
        
        self.visualize = visualize
        
        
        ####################
        # visualize set-up #
        ####################
        if self.visualize:
            plot_start(self.ob, q_s, q_g)

    def continuousMCD(self):
        if not self.is_sol:
            self.J = self.J + self.del_J

        for iters in range(self.n_e):
            optimal_path_to_node = self.expandRoadmap()
        
        for _ in range(5): 
            self.sampleDisplacement(optimal_path_to_node)
        # self.sampleDisplacement(optimal_path_to_node)
        # print(self.D)

        if self.is_new_sol:
            print(self.traj)            
            for _ in range(self.n_o):
                sig = self.localOptimize()
                if not sig:
                    break
            print(self.traj)
            if self.visualize:
                plt.clf()
                plot_start_with_sol(self.q_s,self.q_g)
                plot_obs_with_sol(self.D,self.traj['D'],self.ob)
                for node, node_configuration in self.graph['V'].items():
                    plt.gca().scatter(node_configuration[0],node_configuration[1],c='green', s=10) # ,s=10,alpha=0.3
                    for neighbor_node in self.graph['E'][node]:
                        plt.gca().plot([node_configuration[0],self.graph['V'][neighbor_node][0]],[node_configuration[1],self.graph['V'][neighbor_node][1]],color='orange',markeredgewidth=0.1)
                for i in range(len(self.traj['P'])-1):
                    plt.gca().plot([self.graph['V'][self.traj['P'][i]][0],self.graph['V'][self.traj['P'][i+1]][0]],[self.graph['V'][self.traj['P'][i]][1],self.graph['V'][self.traj['P'][i+1]][1]],color='blue',markeredgewidth=0.1)
                
                
                plot_world()
                
            self.is_new_sol = False
        return self.J
    
    
    #######################
    # expand roadmap func #
    #######################      
    def expandRoadmap(self):
        while True:
            q_d_configuration = sample(self.graph['V'], self.width, self.height, self.ob.static_obstacles)
            q_n, cost_graph = closest(self.graph['V'], self.graph['E'], q_d_configuration, self.J, self.w_l)
            q_configuration, distance = extendToward(self.graph['V'], cost_graph, q_d_configuration, q_n, self.ob.static_obstacles, self.delta, self.J, self.w_l)
            if q_configuration == None:
                continue
            if not neighbors(cost_graph, self.graph['V'], self.graph['E'], self.ob.static_obstacles, q_configuration, distance, self.J, self.w_l, goal_threshold=self.goal_threshold):
                continue
            break
            
        if self.visualize:
            plt.gca().scatter(q_configuration[0],q_configuration[1],c='green', s=10) # ,s=10,alpha=0.3
            for neighbor_node in self.graph['E'][len(self.graph['V'])-1]:
                plt.gca().plot([q_configuration[0],self.graph['V'][neighbor_node][0]],[q_configuration[1],self.graph['V'][neighbor_node][1]],color='orange',markeredgewidth=0.01)
            plot_world()
        
        # plt.show()
        # exit()
        
        best_J, best_displacements, optimal_path_to_node, record_prev_node = discreteMCD_back_checking(self.graph['V'], self.graph['E'], self.D, self.ob.movable_obstacles, self.ob.movable_ob_num, w_l=self.w_l, w_o=self.w_o)

        if best_J:
            if self.J > best_J:
                self.is_sol = True
                record_trajectory(record_prev_node, self.traj, best_displacements)
                self.J = best_J
                self.is_new_sol = True
                
        return optimal_path_to_node

                
    #######################
    # sample displacement #
    #######################
    def sampleDisplacement(self, optimal_path_to_node):
        if self.is_GNN:
            GNN_dice = random.uniform(0,1)
            if GNN_dice < self.lambda_value:
                x=[]
                edge=[]
                for tmp_idx, tmp_ob in enumerate(self.ob.movable_obstacles):
                    x2=[]
                    edge2=[1 for _ in range(len(self.ob.movable_obstacles))]
                    edge2[tmp_idx] = 0
                    for node in self.key_configuration:
                        tmp_point = sg.Point(node[0],node[1])
                        if not tmp_ob.feasibility_check(tmp_point):
                            x2.append(1)
                        else:
                            x2.append(0)
                    x.append(x2)
                    edge.append(edge2)
                x = torch.tensor(x, dtype=torch.float32).cuda()
                edge = torch.tensor(edge, dtype=torch.float32).cuda()
                x=x.unsqueeze(0)
                edge=edge.unsqueeze(0)
                output = self.GNN_model(x,edge).cpu().numpy()
                print(output)
                index_list = []
                for n_idx,i in enumerate(output[0]):
                    if i[0] > self.GNN_threshold:
                        index_list.append(n_idx)
                        
                ob_idx = random.choice(range(len(index_list)))
                ob_idx = index_list[ob_idx]
                
                sample_sig = True
                
                while sample_sig:
                    dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec, axis=0)
                    
                    vec_size = random.uniform(0,1)
                    vec_size *= self.J
                    d = dir_vec * vec_size
                    d = list(d)
                    
                    tmp_x = self.ob.movable_obstacles[ob_idx].x + d[0]
                    tmp_y = self.ob.movable_obstacles[ob_idx].y + d[1]
                    tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=self.ob.movable_obstacles[ob_idx].ob_radius)
                    
                    for static_obstacle in self.ob.static_obstacles:
                        if not static_obstacle.feasibility_check(tmp_ob.ob):
                            continue
                    sample_sig = False
                self.D[ob_idx].append([d,vec_size])
                self.D[ob_idx].sort(key=lambda x:x[1])
            else:
                if self.is_sol:
                    dice = random.uniform(0,1)
                    if dice < 0.5:
                        unblocking(self.D, optimal_path_to_node, self.graph['V'], self.graph['E'], self.ob.movable_obstacles, self.ob.static_obstacles, self.J, self.w_l, self.w_o)
                    else:
                        refining(self.D, optimal_path_to_node, self.ob.movable_obstacles, self.ob.static_obstacles, self.ob.movable_ob_num, self.J) #, self.ob.static_obstacles, self.J)
                else:
                    unblocking(self.D, optimal_path_to_node, self.graph['V'], self.graph['E'], self.ob.movable_obstacles, self.ob.static_obstacles, self.J, self.w_l, self.w_o)
        else:
            if self.is_sol:
                dice = random.uniform(0,1)
                if dice < 0.5:
                    unblocking(self.D, optimal_path_to_node, self.graph['V'], self.graph['E'], self.ob.movable_obstacles, self.ob.static_obstacles, self.J, self.w_l, self.w_o)
                else:
                    refining(self.D, optimal_path_to_node, self.ob.movable_obstacles, self.ob.static_obstacles, self.ob.movable_ob_num, self.J) #, self.ob.static_obstacles, self.J)
            else:
                unblocking(self.D, optimal_path_to_node, self.graph['V'], self.graph['E'], self.ob.movable_obstacles, self.ob.static_obstacles, self.J, self.w_l, self.w_o)
                
                
    ######################
    # local optimization #
    ######################
    def localOptimize(self, threshold=0.1):
        optimize_path_only(self.graph['V'], self.graph['E'], self.traj, self.D, self.traj['D'], self.ob.movable_obstacles)
        optimize_displacements(self.graph['V'], self.graph['E'], self.traj, self.D, self.traj['D'], self.ob.movable_obstacles, threshold)
        tmp_J = cal_J(self.graph['V'], self.traj, self.D, self.w_l, self.w_o)
        if self.J!=tmp_J:
            self.J=tmp_J
            return True
        else:
            return False
        
        
    ##################
    # util functions #
    ##################
    def reset(self):
        vertices = dict()
        vertices[0] = np.array(self.graph['V'][0])
        vertices[1] = np.array(self.graph['V'][1])
        edge = dict()
        edge[0] = []
        edge[1] = []
        
        self.graph = dict()
        self.graph['V'] = vertices
        self.graph['E'] = edge
        
        self.D = [[[0.0 for _ in range(self.dimension)]] for _ in range(self.ob_num)]

        self.traj = dict()
        self.traj['P'] = [0,1]
        self.traj['D'] = [[0.0 for _ in range(self.dimension)] for _ in range(self.ob_num)]
        self.J = 0 # set to 0 for planning
        
        self.is_sol = False
        self.is_new_sol = False
        
        self.unblocking_obstacles_set = []
        self.refining_obstacles_count = [0 for _ in range(self.ob_num)]
        
        self.data_set = []
    
    def save_dataset(self, file_name):
        current_path = os.getcwd()
        data_path = current_path+'/data/'+file_name
        if os.path.isfile(os.path.join('./data',file_name)):
            self.load_dataset(data_path)
        self.data_set = list(self.data_set) 
        self.data_set.append([self.graph,self.D,self.traj,self.ob,self.J,self.is_sol])
        np.save(data_path,np.array(self.data_set))

    def load_dataset(self, path):
        self.data_set = np.load(path,allow_pickle=True)

    
        
if __name__=='__main__':
    print("MCD !!")
    
    
    
    
    
    
    
    
    
    
    
    

            


    
