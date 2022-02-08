import math
import random

import numpy as np
import shapely.geometry as sg

from Utils import dist, visible, dijstra_J_reachable
from pq import PriorityQueue
from Object import Obstacle

# sample
def sample(graph, width, height, static_obstacles, threshold=0.001):
    while True:
        q_d = (np.random.rand(2) - 0.5)
        q_d[0] *= width
        q_d[1] *= height
        
        q_d_shapely = sg.Point(q_d[0], q_d[1])
        
        for v in graph:
            if dist(graph[v], q_d) < threshold:
                continue
            
        for obstacle in static_obstacles:
            if not obstacle.feasibility_check(q_d_shapely):
                continue
        
        break
    return q_d.tolist()

# closest
def closest(graph_v, graph_e, q_d_configuration, J, w_l):
    cost_graph = dijstra_J_reachable(graph_v,graph_e,J,w_l)
    for n, q_n_idx in enumerate(cost_graph):
        if n==0:
            q_n = q_n_idx
            q_n_configuration = graph_v[q_n]
        else:
            if dist(graph_v[q_n_idx],q_d_configuration) < dist(q_n_configuration,q_d_configuration):
                q_n = q_n_idx
                q_n_configuration = graph_v[q_n_idx]
    return q_n, cost_graph

# extendToward
def extendToward(graph_v, cost_graph, q_d_configuration, q_n, static_obstacles, delta, J, w_l, threshold=0.0001):
    q_n_configuration = graph_v[q_n]
    q_configuration = list(q_n_configuration)
    delta_multiplier = min(delta/dist(q_n_configuration,q_d_configuration),1)
    q_configuration[0] += ((q_d_configuration[0] - q_configuration[0]) * delta_multiplier)
    q_configuration[1] += ((q_d_configuration[1] - q_configuration[1]) * delta_multiplier)
    
    for static_obstacle in static_obstacles:
        if not visible(q_n_configuration,q_configuration,static_obstacle):
            return None, None

    for v in graph_v:
        if dist(graph_v[v], q_configuration) < threshold:
            return None, None
    
    if (w_l * dist(q_configuration, q_n_configuration)) < (J-cost_graph[q_n]):
        return q_configuration, dist(q_configuration, q_n_configuration)
    else:
        return None, None

# neighbors
def neighbors(cost_graph, graph_v, graph_e, static_obstacles, q_configuration, distance, J, w_l, goal_threshold=5.0):
    e = 0
    for i in graph_e:
        e += len(graph_e[i])
    m = (1 + 1/distance) * e * math.log(len(graph_v)) + 1
    
    count = int(m)
    # count = 3
    
    tmp = list()
    for i in cost_graph:
        sig = False
        for static_obstacle in static_obstacles:
            if not visible(graph_v[i],q_configuration,static_obstacle):
                sig = True
                break
        if sig:
            continue
        tmp_distance = dist(graph_v[i], q_configuration)
        if (w_l*tmp_distance+cost_graph[i]) < J:
            tmp.append([i,tmp_distance])
    
    # for checking #
    if count > 10:
        count=10
    ################
        
    if len(tmp)==0:
        return False
    else:
        q = len(graph_v)
        graph_v[q] = q_configuration
        graph_e[q] = []
        
    tmp.sort(key=lambda x:x[1])
    
    for i, tmp_distance in tmp:
        if count > 0:
            graph_e[i].append(q)
            graph_e[q].append(i)
            count -= 1
            
    if dist(graph_v[1],q_configuration) < goal_threshold:
        if not q in graph_e[1]:
            graph_e[q].append(1)
            graph_e[1].append(q)
    
    return True
    
# discreteMCD
def discreteMCD_back_checking(graph_v, graph_e, D, movable_obstacles, ob_num, w_l=0.1, w_o=0.1, threshold=0.001):
    queue = PriorityQueue()
    queue.push(0.0,[0,[0 for _ in range(ob_num)],None]) # cost, node_num, displacement index, parent node
    finished_node = list()
    record_prev_node = dict()
    optimal_path_to_node = list()
    
    while queue.size()!=0:
        cost, info = queue.pop()
        node_num = info[0]
        optimal_displacement_indexes = info[1]
        prev_node = info[2]
        if not node_num in finished_node:
            record_prev_node[node_num] = prev_node
            finished_node.append(node_num)
            optimal_path_to_node.append([node_num, cost, list(optimal_displacement_indexes)])
            
            if node_num==1:
                return cost, optimal_displacement_indexes, optimal_path_to_node, record_prev_node

            for e in graph_e[node_num]:
                if not e in finished_node:
                    tmp_optimal_dispalcement_indexes = list(optimal_displacement_indexes)
                    is_ok = True
                    new_cost = cost
                    for ob_num, movable_obstacle in enumerate(movable_obstacles):
                        ob_d_idx = tmp_optimal_dispalcement_indexes[ob_num]
                        ob_priority = D[ob_num][ob_d_idx][1]
                        is_ok2 = False
                        for tmp_ob_num, (d_sample, tmp_priority) in enumerate(D[ob_num][ob_d_idx:]):
                            tmp_x = movable_obstacle.x + d_sample[0]
                            tmp_y = movable_obstacle.y + d_sample[1]
                            tmp_ob = Obstacle(init_position=(tmp_x,tmp_y), radius=movable_obstacle.ob_radius)
                            if visible(graph_v[node_num],graph_v[e],tmp_ob):
                                tmp_optimal_dispalcement_indexes[ob_num] = tmp_ob_num + ob_d_idx
                                is_ok2 = True
                                new_cost += (tmp_priority - ob_priority)
                                if ob_priority==0 and tmp_priority!=0:
                                    new_cost+=w_o
                                break
                        if not is_ok2:
                            is_ok=False
                            break
                    if is_ok:
                        new_cost += w_l*dist(graph_v[node_num], graph_v[e])
                        queue.push(new_cost,[e,tmp_optimal_dispalcement_indexes,node_num])
    return None, None, optimal_path_to_node, None


if __name__=="__main__":
    graph_v = dict()
    graph_v[0] = [-10.0,0.0]
    graph_v[1] = [10.0,0.0]
    graph_v[2] = [-6.0,0.0]
    graph_v[3] = [-3.0,0.0]
    graph_v[4] = [3.0,0.0]
    graph_v[5] = [6.0,0.0]
    graph_v[6] = [0.0,0.0]
    graph_v[7] = [-4.5,0.5]
    graph_v[8] = [-1.5,0.5]
    graph_v[9] = [2.5,0.5]
    
    graph_e = dict()
    graph_e[0] = [2]
    graph_e[1] = [5]
    graph_e[2] = [0,3,7]
    graph_e[3] = [2,6,7,8]
    graph_e[4] = [5,6]
    graph_e[5] = [1,4,9]
    graph_e[6] = [3,4]
    graph_e[7] = [2,3]
    graph_e[8] = [3,9]
    graph_e[9] = [6,8]

    tmp_obstacle = Obstacle(init_position=(0.0,-0.25), radius=0.5)
    movable_obstacles = [tmp_obstacle]
    
    D = [[[[0.0,0.0],0.0], [[0.0,-0.26],0.26]]]

    ob_num = 1
        
    _, d, answer, r = discreteMCD_back_checking(graph_v, graph_e, D, movable_obstacles, ob_num, w_l=0.1, w_o=0.1, threshold=0.001)
    
    print(d)
    print(answer)
    print(r)