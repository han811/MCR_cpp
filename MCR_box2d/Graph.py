from typing import List, Set

class Node:
    
    def __init__(self, q: list, index: int, k: int):
        
        # configuration and index
        self.q: list = q
        self.index: int = index


class Edge:

    def __init__(self, index: int):
        
        # adjancy list index
        self.index: int = index
        self.neighbors: list = list()

    def addNeighbor(self, index: int):

        if index in self.neighbors:
            print('you are setting same neighbot twice!!')
        else:
            self.neighbors.append(index)


class Graph:
    
    def __init__(self, q_s, q_g):
        
        # Node list and Edge list
        self.nodes: list = [Node(q_s,0), Node(q_g,1)]
        self.edges: list = []
        
        # set for next index
        self.next_index = 2