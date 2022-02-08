'''
collision check by shapely
feasibility check by shapely
'''
import math
import random

import numpy as np
from matplotlib import pyplot as plt
import shapely.geometry as sg

from pq import PriorityQueue
from Object import Obstacle

# MCD utils
def dist(q1, q2):
    return math.sqrt((q1[0] - q2[0])**2 + (q1[1] - q2[1])**2)
            
def visible(c1, c2, movable_obstacle):
    line = [tuple(c1), tuple(c2)]
    tmp_line = sg.LineString(line)
    return movable_obstacle.feasibility_check(tmp_line)

def dijstra_J_reachable(graph_v, graph_e, J, w_l):
    queue = PriorityQueue()
    queue.push(0.0,0) # cost, node
    finished_node = set()
    cost_graph = dict()

    while queue.size()!=0:
        cost, node = queue.pop()
        if not node in finished_node:
            finished_node.add(node)
            cost_graph[node] = cost
            for e in graph_e[node]:
                if not e in finished_node:
                    new_cost = cost + w_l * dist(graph_v[e], graph_v[node])
                    if new_cost < J:
                        queue.push(new_cost,e)
    return cost_graph

def record_trajectory(record_prev_node, traj, optimal_D):
    node = 1
    best_traj = [1]
    while node!=0:
        prev_node = record_prev_node[node]
        best_traj = [prev_node] + best_traj
        node = prev_node
    traj['P'] = list(best_traj)
    traj['D'] = list(optimal_D)    

def cal_J(graph_v, traj, D, w_l, w_o):
    cost = 0.0
    
    for i in range(len(traj['P'])-1):
        cost += w_l * dist(graph_v[traj['P'][i]],graph_v[traj['P'][i+1]])
    
    for d_idx, d in enumerate(traj['D']):
        if D[d_idx][d][1]!=0.0:
            cost += D[d_idx][d][1]
            cost += w_o
    
    return cost
    
# plot utils
def plot_start(ob, q_s, q_g):
    print("visualization start !!!!")
    ob.plot()
    plt.gca().scatter(q_s[0],q_s[1],c='red') # ,s=10,alpha=0.3
    plt.gca().scatter(q_g[0],q_g[1],c='blue') # ,s=10,alpha=0.3
    plot_world()

def plot_start_with_sol(q_s, q_g):
    print("visualization start !!!!")
    plt.gca().scatter(q_s[0],q_s[1],c='red') # ,s=10,alpha=0.3
    plt.gca().scatter(q_g[0],q_g[1],c='blue') # ,s=10,alpha=0.3
    plot_world()

def plot_obs_with_sol(D,optimal_displacements,ob):
    for obstacle in ob.static_obstacles:
        obstacle.plot()
    for idx,i in enumerate(optimal_displacements):
        if i==0:
            ob.movable_obstacles[idx].plot()
        else:
            ob.movable_obstacles[idx].plot(moved=True,check=False)
            new_x = ob.movable_obstacles[idx].x + D[idx][i][0][0]
            new_y = ob.movable_obstacles[idx].y + D[idx][i][0][1]
            new_ob = Obstacle(init_position=(new_x,new_y), radius=ob.movable_obstacles[idx].ob_radius)
            new_ob.plot(moved=True,check=True)
    
def plot_world():
    plt.axis("equal")
    plt.pause(0.001)
    plt.grid(True)
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

def plot_sol(traj, graph_v):
    for i in range(len(traj['P'])-1):
        plt.gca().plot([graph_v[traj['P'][i]][0],graph_v[traj['P'][i+1]][0]],[graph_v[traj['P'][i]][1],graph_v[traj['P'][i+1]][1]],color='blue',markeredgewidth=0.1)
        print(graph_v[traj['P'][i]][0])
    
    
if __name__=="__main__":
    # dijstra algorithm test

    test_graph_v = dict()
    
    test_graph_v[0] = (0.0,0.0)
    test_graph_v[1] = (1.0,0.0)
    test_graph_v[2] = (0.0,1.0)
    test_graph_v[3] = (1.0,1.0)
    test_graph_v[4] = (2.0,1.0)
    
    test_graph_e = dict()
    
    test_graph_e[0] = [1,2]
    test_graph_e[1] = [0,3,4]
    test_graph_e[2] = [0,3]
    test_graph_e[3] = [1,2,4]
    test_graph_e[4] = [1,3]
    
    print(dijstra_J_reachable(test_graph_v,test_graph_e,2.1))