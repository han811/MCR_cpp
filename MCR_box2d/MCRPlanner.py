import random
import math

from Graph import Node, Edge, Graph
from Roadmap import Roadmap
from Env import World

class MCRPlanner:
    
    def __init__(self, world, graph, delta):
        self.world = world
        self.graph = graph
        
        self.delta = delta
        self.k = self.graph.nodes[0].n_cover + self.graph.nodes[1].n_cover


    def sample(self):
        q_w = random.random() * self.world.width
        q_h = random.random() * self.world.height
        
        return [q_w, q_h]

    def closest(self, graph, k, q_d):
        closest_node = None
        closest_distance = math.inf
        for node in graph:
            if node.n_cover <= k and self.l2_distance(node.q, q_d) <= closest_distance:
                closest_distance = self.l2_distance(node.q, q_d)
                closest_node = node
        
        return closest_node

    def extendToward(self, q_node, q_d, delta, k):
        tmp_k = q_n.


    def l2_distance(self, q1, q2):
        diff = []
        for i in range(self.World.q_dim):
            diff.append(q1[i] - q2[i])
        
        square_sum = 0
        for d in diff:
            square_sum += math.pow(d,2)

        return math.sqrt(square_sum) 
            
            
