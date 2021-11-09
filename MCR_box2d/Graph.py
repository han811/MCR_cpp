import math

class Node:
    def __init__(self, q, index, cover):
        self.q = q
        self.index = index
        self.cover = cover
        self.n_cover = len(cover)

class Edge:
    def __init__(self, index):
        self.from_index = index
        self.to_index = list()
        self.to_distances = list()

    def addIndex(self, index, distance):
        self.to_index.append(index)
        self.to_distances.append(distance)

    def size(self):
        return len(self.to_index)


class Graph:

    def __init__(self, q_s, q_g):
        # initial setting for graph - start and goal configurations
        self.q_s = Node(q_s,0,[])
        self.q_g = Node(q_g,1,[])
        self.v_num = 2

        # setting for nodes and edges
        self.nodes = [Node(q_s,0), Node(q_g,1)] # node list
        self.edges = [Edge(0), Edge(1)] # from list , to list adjancy matrix

    # add and getter of Node
    def addNode(self, q):
        self.nodes.append(Node(q,self.v_num))
        self.v_num += 1

    def addEdge(self, index1, index2):
        distance = 0
        for i in range(len(self.nodes[index1].q)):
            distance += math.pow(self.nodes[index1].q[i] - self.nodes[index2].q[i], 2)
        distance = math.sqrt(distance)

        self.edges[index1].addIndex(index2, distance)
        self.edges[index2].addIndex(index1, distance)

    def getNode(self, index):
        return self.nodes[index]



    

