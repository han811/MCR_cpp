import math
import random

import numpy as np

from Utils import dist, visible
from Object import Obstacle

# unblocking obstacle check
def unblocking_obstacle_check(optimal_path_to_node, graph_v, graph_e, movable_obstacles, J, w_l, w_o):
    edge_check_list = set()
    finished_node = []
    
    unblocking_set = dict()
    for node, cost, _ in optimal_path_to_node:
        for e in graph_e[node]:
            if not ((node,e) in edge_check_list):
                new_cost = cost + w_l * dist(graph_v[node], graph_v[e])
                if new_cost < J:
                    edge_check_list.add((node,e))
                    edge_check_list.add((e,node))
                    for ob_idx, obstacle in enumerate(movable_obstacles):
                        if not visible(graph_v[node], graph_v[e], obstacle):
                            if ob_idx in unblocking_set:
                                if unblocking_set[ob_idx] < (J-new_cost):
                                    unblocking_set[ob_idx] = (J-new_cost)
                            else:
                                unblocking_set[ob_idx] = J-new_cost
    return unblocking_set

# unblocking
def unblocking(D, optimal_path_to_node, graph_v, graph_e, movable_obstacles, static_obstacles, J, w_l, w_o):
    unblocking_set = unblocking_obstacle_check(optimal_path_to_node, graph_v, graph_e, movable_obstacles, J, w_l, w_o)
    # print(unblocking_set)
    size = len(unblocking_set)
    if size==0:
        return
    ob_idx = random.choice(range(size))
    ob_idx_list = list(unblocking_set.keys())
    ob_idx = ob_idx_list[ob_idx]
    cost = unblocking_set[ob_idx]
    
    sample_sig = True
    
    while sample_sig:
        dir_vec = np.random.randn(2)
        dir_vec /= np.linalg.norm(dir_vec, axis=0)
        
        vec_size = random.uniform(0,1)
        vec_size *= cost
        
        d = dir_vec * vec_size
        d = list(d)
        
        tmp_x = movable_obstacles[ob_idx].x + d[0]
        tmp_y = movable_obstacles[ob_idx].y + d[1]
        tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=movable_obstacles[ob_idx].ob_radius)
        
        for static_obstacle in static_obstacles:
            if not static_obstacle.feasibility_check(tmp_ob.ob):
                continue
        sample_sig = False
        
    D[ob_idx].append([d,vec_size])
    D[ob_idx].sort(key=lambda x:x[1])

# refining    
def refining(D, optimal_path_to_node, movable_obstacles, static_obstacles, ob_num, J):
    bins = [0 for _ in range(ob_num)]
    sample_range = [0 for _ in range(ob_num)]
    for node, cost, d in optimal_path_to_node:
        for d_idx, tmp_d in enumerate(d):
            if D[d_idx][tmp_d][0][0]!=0.0 or D[d_idx][tmp_d][0][1]!=0.0:
                bins[d_idx] += 1
                if (J - cost) > sample_range[d_idx]:
                    sample_range[d_idx] = (J - cost)
    total = sum(bins) + 1e-10
    for idx in range(len(bins)):
        bins[idx] /= total
    
    rand_prob = random.uniform(0,1)
    cumul_prob = 0
    sample_idx = -1
    for idx in range(len(bins)):
        cumul_prob += bins[idx]
        if cumul_prob>rand_prob:
            sample_idx = idx
            break
        
    if sample_idx!=-1:
        sample_sig = True
    
        while sample_sig:
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec, axis=0)
            
            vec_size = random.uniform(0,1)
            vec_size *= cost
            
            d = dir_vec * vec_size
            d = list(d)
            
            tmp_x = movable_obstacles[sample_idx].x + d[0]
            tmp_y = movable_obstacles[sample_idx].y + d[1]
            tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=movable_obstacles[sample_idx].ob_radius)
            
            for static_obstacle in static_obstacles:
                if not static_obstacle.feasibility_check(tmp_ob.ob):
                    continue
            sample_sig = False
            
            
        D[sample_idx].append([d,vec_size])
        D[sample_idx].sort(key=lambda x:x[1])
        