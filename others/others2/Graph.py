from typing import List, Set

class Node:
    
    def __init__(self, q: list, index: int, k: int, cover: set = set()):
        
        # configuration and index
        self.q: list = q
        self.index: int = index

        self.k = k
        self.cover = cover
        self.s_cover = set()

        self.parent = None


class Edge:

    def __init__(self, index: int):
        
        # adjancy list index
        self.index: int = index
        self.neighbors: list = list()
        self.distance: list = list()
        self.covers: list = list()

    def addNeighbor(self, index: int, d: float, cover: set):

        if index in self.neighbors:
            print(index)
            print(self.index)
            print(self.neighbors)
            print('you are setting same neighbot twice!!')
            exit()
        else:
            self.neighbors.append(index)
            self.distance.append(d)
            self.covers.append(cover)


class Graph:
    
    def __init__(self, q_s, q_g):
        
        # Node list and Edge list
        self.nodes: list = [Node(q_s,0,0,set()), Node(q_g,1,0,set())]
        self.edges: list = [Edge(0), Edge(1)]
        
        # set for next index
        self.next_index = 2