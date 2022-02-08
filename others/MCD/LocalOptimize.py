import math
import random

import numpy as np

from Object import Obstacle
from Utils import dist, visible

# optimize path
def optimize_path_only(graph_v, graph_e, traj, D, optimal_displacements, movable_obstacles):
    sig = True
    while sig:
        traj_length = len(traj['P'])
        sig2 = False
        for i1 in range(traj_length-2):
            if sig2:
                break
            for i2 in range(i1+2,traj_length):
                if sig2:
                    break
                is_visible = True
                for ob_idx, obstacle in enumerate(movable_obstacles):
                    tmp_x = obstacle.x + D[ob_idx][optimal_displacements[ob_idx]][0][0]
                    tmp_y = obstacle.y + D[ob_idx][optimal_displacements[ob_idx]][0][1]
                    tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=obstacle.ob_radius)
                    if not visible(graph_v[traj['P'][i1]], graph_v[traj['P'][i2]], tmp_ob):
                        is_visible = False
                        break
                if is_visible:
                    check_node_i1 = traj['P'][i1]
                    check_node_i2 = traj['P'][i2]
                    traj['P'] = traj['P'][:i1+1] + traj['P'][i2:]
                    if not check_node_i2 in graph_e[check_node_i1]:
                        graph_e[check_node_i1].append(check_node_i2)
                        graph_e[check_node_i2].append(check_node_i1)
                    sig2 = True
        if not sig2:
            sig=False
                    
# optimize displacement     
def optimize_displacements(graph_v, graph_e, traj, D, optimal_displacements, movable_obstacles, threshold=0.01):
    for d_idx, d in enumerate(optimal_displacements):
        d_size = D[d_idx][d][1]
        if d_size!=0.0:
            d_direction_vec = [D[d_idx][d][0][0]/d_size, D[d_idx][d][0][1]/d_size]
            d_direction_vec = np.array(d_direction_vec)
            
            right_d = d_size
            left_d = 0
            
            while (right_d-left_d)>threshold:
                sig = True
                mid_d = (left_d+right_d)/2
                tmp_d = d_direction_vec * mid_d
                
                tmp_x = movable_obstacles[d_idx].x+tmp_d[0]
                tmp_y = movable_obstacles[d_idx].y+tmp_d[1]
                tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=movable_obstacles[d_idx].ob_radius)
                
                for i in range(len(traj['P'])-1):
                    if not visible(graph_v[traj['P'][i]], graph_v[traj['P'][i+1]], tmp_ob):
                        sig = False
                
                if sig:
                    right_d = mid_d
                else:
                    left_d = mid_d
            
            final_d = d_direction_vec * right_d
            final_d = list(final_d)
            D[d_idx].append([final_d,right_d])
            D[d_idx].sort(key=lambda x:x[1])
            new_d_idx = -1
            for idx, sample in enumerate(D[d_idx]):
                if sample[1]==right_d:
                    new_d_idx = idx
                    break
            # print(D)
            # print(d_idx)
            # print(new_d_idx)
            optimal_displacements[d_idx] = new_d_idx
    print("optimal_displacements:",optimal_displacements)
