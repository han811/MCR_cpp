import os, sys
import argparse

import torch
import torch.utils.data as data

class Graph(data.Dataset):
    def __init__(self, cuda):
        self.graph = []
        self.adjancy_matrix = []
        self.label = []
        self.cuda = cuda

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        if self.cuda:
            new_graph = list()
            for i in self.graph[index]:
                new_graph.append(i.cuda())
            return new_graph, self.adjancy_matrix[index].cuda(), self.label[index].cuda()
        else:
            return self.graph[index], self.adjancy_matrix[index], self.label[index]

    def add_graph(self, x, adjancy_matrix, label):
        tmp_graph = list()
        
        # for node in list(x.values()):
        for node in x:
            tmp_graph.append(torch.tensor([node], dtype=torch.float32))
        
        tmp_adjancy_matrix = torch.tensor(adjancy_matrix, dtype=torch.float32)
        tmp_label = torch.tensor(label, dtype=torch.float32)
        self.graph.append(tmp_graph)
        self.adjancy_matrix.append(tmp_adjancy_matrix)
        self.label.append(tmp_label)

if __name__=='__main__':
    g = Graph(False)
    x = dict()
    x[0] = [3,2]
    x[1] = [2,4]
    y = [0,1]
    edge = [[0,1],[1,0]]
    g.add_graph(x, edge, y)
    print(g.graph)
    print(g.adjancy_matrix)
    print(g.label)
    print(g[0])
