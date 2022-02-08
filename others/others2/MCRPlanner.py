import heapq
import math
import random
from typing import List, Dict, Tuple


from Graph import Node, Edge, Graph
from World import World

class MCRPlanner:
    
    def __init__(self, world: World, graph: Graph, delta: float, m: int, N_raise: int):

        # initial world and graph
        self.world: World = world
        self.graph: Graph = graph

        # S_min and k
        self.S_min: set = set()
        
        self.graph.nodes[0].k = len(self.graph.nodes[0].cover)
        self.graph.nodes[0].cover = self.world.checkPointCover(self.world.q_s, range(len(self.world.obstacles)))
        self.graph.nodes[0].s_cover = set(self.graph.nodes[0].cover)

        self.graph.nodes[1].k = len(self.graph.nodes[1].cover)
        self.graph.nodes[1].cover = self.world.checkPointCover(self.world.q_s, range(len(self.world.obstacles)))
        self.graph.nodes[1].s_cover = set(range(len(self.world.obstacles)+1))


        self.K = len(self.graph.nodes[0].cover | self.graph.nodes[1].cover)

        self.N_raise = N_raise

        # extend toward step size delta
        self.delta: float = delta
        self.m = m

        # indicator to notice whether there is at least a trajectory or not
        self.is_traj = False
        self.traj = list()
            

    def sample(self):

        q_d: list = [0 for _ in range(self.world.dim_config)]
        for idx in range(self.world.dim_config):
            q_d[idx] = random.random() * self.world.world_size[idx]
        return q_d

    def closest(self, nodes, k, q_d):
        
        q_n = None
        closest_distance = float('inf')
        node_idx = 0

        for idx, node in enumerate(nodes):
            if (len(node.s_cover) <= k) and (self.l2_distance(node.q, q_d) <= closest_distance):
                closest_distance = self.l2_distance(node.q, q_d)
                q_n = node
                node_idx = idx
        return q_n, node_idx

    def extendToward(self, q_node, q_d, delta, k):
        
        distance = self.l2_distance(q_node.q, q_d)
        direction_vector = list(q_node.q)
        
        for idx, qs in enumerate(list(zip(q_node.q,q_d))):
            direction_vector[idx] = (qs[1] - qs[0]) / distance
        
        if distance > delta:
            q_prime = list(q_node.q)
            for idx, coordinate in enumerate(direction_vector):
                q_prime[idx] += coordinate * delta
        else:
            q_prime = list(q_d)
        
        for step in range(5):
            if step != 0:
                for idx in range(len(q_prime)):
                    q_prime[idx] = q_node.q[idx] + (q_prime[idx] - q_node.q[idx]) * 0.5
            sig = True
            for obstacle in self.world.obstacles:
                if obstacle.is_static and (obstacle.pointCollisionCheck(q_prime,self.world.transform) or obstacle.segmentCollisionCheck(q_prime,q_node.q,self.world.transform,self.world.step)):
                    sig = False
            if sig:
                S = q_node.cover | self.world.checkPointCover(q_prime, range(len(self.world.obstacles))) | self.world.checkLineCover(q_node.q, q_prime, range(len(self.world.obstacles)))
                if len(S) < k:
                    return q_prime, S, self.world.checkLineCover(q_node.q, q_prime, range(len(self.world.obstacles)))
        return None, None, None

    def neighbors(self, graph, q, k):
        
        neighbor_nodes = list()
        que = []

        for node in graph.nodes:
            S = node.cover | self.world.checkPointCover(q, range(len(self.world.obstacles))) | self.world.checkLineCover(node.q, q, range(len(self.world.obstacles)))
            if len(S) < k:
                heapq.heappush(que, (self.l2_distance(node.q, q), node.index))
        
        for idx in range(self.m):
            if idx==0:
                heapq.heappop(que)[1]
                continue
            if not que:
                break
            neighbor_nodes.append(heapq.heappop(que)[1])

        return neighbor_nodes

    def l2_distance(self, q1, q2):

        diff = []
        for i in range(self.world.dim_config):
            diff.append(q1[i] - q2[i])
        
        square_sum = 0
        for d in diff:
            square_sum += d * d

        return math.sqrt(square_sum) 


    def expandRoadmap(self):
        q_d = self.sample()
        q_node, q_node_idx = self.closest(self.graph.nodes, self.K, q_d)

        q_prime, q_prime_cover, edge_cover = self.extendToward(q_node, q_d, self.delta, self.K)
        if q_prime:
            neighbor_nodes = self.neighbors(self.graph, q_prime, self.K)
            self.graph.nodes.append(Node(q_prime, self.graph.next_index, len(q_prime_cover), q_prime_cover))
            self.graph.edges.append(Edge(self.graph.next_index))
            self.graph.edges[q_node_idx].addNeighbor(self.graph.next_index, self.l2_distance(q_prime,q_node.q), edge_cover)
            self.graph.edges[self.graph.next_index].addNeighbor(q_node_idx, self.l2_distance(q_prime,q_node.q), edge_cover)

            for node_idx in neighbor_nodes:
                if self.l2_distance(self.graph.nodes[node_idx].q, q_prime) < self.delta:
                    self.graph.edges[node_idx].addNeighbor(self.graph.next_index, self.l2_distance(q_prime,self.graph.nodes[node_idx].q), self.world.checkLineCover(q_prime,self.graph.nodes[node_idx].q,range(len(self.world.obstacles))))
                    self.graph.edges[self.graph.next_index].addNeighbor(node_idx, self.l2_distance(q_prime,self.graph.nodes[node_idx].q), self.world.checkLineCover(q_prime,self.graph.nodes[node_idx].q,range(len(self.world.obstacles))))
                   
            self.graph.next_index += 1

            return True
        else:
            return False

    def computeMinimumExplanation(self):
        queue = []
        ks = {node: len(self.world.obstacles)+1 for node in range(len(self.graph.nodes))}
        distances = {node: float('inf') for node in range(len(self.graph.nodes))}
        heapq.heappush(queue, (self.graph.nodes[0].k, 0.0, 0, set(self.graph.nodes[0].s_cover)))
        ks[0] = 0
        distances[0] = 0.0
        while queue:
            current_k, current_distance, current_node, current_s_cover = heapq.heappop(queue)
            if ks[current_node] < current_k or distances[current_node] < current_distance:
                continue
            for idx, neighbor_idx in enumerate(self.graph.edges[current_node].neighbors):
                new_s_cover = current_s_cover | self.graph.nodes[neighbor_idx].cover | self.graph.edges[current_node].covers[idx]
                new_k = len(new_s_cover)
                new_distance = current_distance + self.graph.edges[current_node].distance[idx]
                if new_k <= ks[neighbor_idx] and new_distance < distances[neighbor_idx]:
                    ks[neighbor_idx] = new_k
                    distances[neighbor_idx] = new_distance
                    self.graph.nodes[neighbor_idx].parent = current_node
                    self.graph.nodes[neighbor_idx].s_cover = new_s_cover
                    heapq.heappush(queue, (new_k, new_distance, neighbor_idx, new_s_cover))
        
        if distances[1]!=float('inf'):
            self.is_traj = True
            tmp_traj = list()
            tmp_traj.append(0)
            while True:
                tmp_node_idx = tmp_traj[0]
                tmp_node_parent = self.graph.nodes[tmp_node_idx].parent
                tmp_traj.append(tmp_node_parent)
                if tmp_node_parent == 0:
                    break
            self.S_min = set(self.graph.nodes[1].s_cover)


    def continuousMCR(self, N_iter):
        for it in range(N_iter):
            print('iteration step:',it)
            if self.expandRoadmap():
                self.computeMinimumExplanation()
            if it % self.N_raise == 0:
                self.K += 1
            if self.is_traj:
                if self.K >= len(self.S_min):
                    self.K = len(self.S_min) - 1
