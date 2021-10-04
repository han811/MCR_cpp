import argparse
import os, sys
from typing import Any, List, Dict, Set

import torch
import torch.utils.data as data

class Graph(data.Dataset):
    '''My custom graph dataset structure'''
    def __init__(self, graph_nodes: list = list(), adjancy_matrices: list = list(), labels: list = list()) -> None:
        '''
            nodes : obstacle graph nodes represented using key configurations
            adjancy_matrices : fully connected adjancy matrices
            labels : label information removed obstacles
        '''
        self.graph_nodes = graph_nodes
        self.adjancy_matrices = adjancy_matrices
        self.labels = labels

    def __len__(self):
        return len(self.graph_nodes)
    
    def __getitem__(self, index):
        graph_node = torch.FloatTensor(self.graph_nodes[index])
        adjancy_matrix = torch.FloatTensor(self.adjancy_matrices[index])
        label = torch.FloatTensor(self.labels[index])
        return graph_node, adjancy_matrix, label

    def empty(self):
        self.graph_nodes.clear()
        self.adjancy_matrices.clear()
        self.labels.clear()

    def add_graph(self, x, adjancy_matrix, label):
        self.graph_nodes.append(x)
        self.adjancy_matrices.append(adjancy_matrix)
        self.labels.append(label)

if __name__=='__main__':
    g = Graph()
    for i in range(4):
        x: list = list()
        x.append([3+i,2+i,4+i,1+i])
        x.append([2+i,4+i,1+i,5+i])
        x.append([6+i,1+i,3+i,2+i])
        y = [0,1,1]
        edge = [[0,1,1],[1,0,1],[1,1,0]]
        g.add_graph(x, edge, y)

    dataloader = data.DataLoader(g, batch_size=2, shuffle=True)
    print(g.graph_nodes)
    print(g.adjancy_matrices)
    print(g.labels)
    print(g[0])


