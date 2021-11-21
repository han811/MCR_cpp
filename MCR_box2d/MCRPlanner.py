import math
import random
from typing import List, Dict, Tuple

from Graph import Node, Edge, Graph
from World import World

class MCRPlanner:
    
    def __init__(self, world: World, graph: Graph, delta: float):

        # initial world and graph
        self.world: World = world
        self.graph: Graph = graph

        # S_min and k
        self.S_min: set = set()
        self.k = len(self.world.checkPointCover(self.world.q_s, range(len(self.world.obstacles))) | self.world.checkPointCover(self.world.q_g, range(len(self.world.obstacles))))

        # extend toward step size delta
        self.delta: float = delta


    def sample(self):

        q_d: list = list()
        for idx in range(self.world.dim_config):
            q_d.append(random.random() * self.world.world_size[idx])
        
        return q_d

    def closest(self, nodes, k, q_d):
        
        q_n = None
        closest_distance = math.inf
        for node in nodes:
            if node.n_cover <= k and self.l2_distance(node.q, q_d) <= closest_distance:
                closest_distance = self.l2_distance(node.q, q_d)
                q_n = node
        
        return q_n

    def extendToward(self, q_node, q_d, delta, k):
        tmp_k = q_n.


    def l2_distance(self, q1, q2):

        diff = []
        for i in range(self.world.dim_config):
            diff.append(q1[i] - q2[i])
        
        square_sum = 0
        for d in diff:
            square_sum += d * d

        return math.sqrt(square_sum) 
            
            
