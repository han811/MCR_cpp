import os
import sys

class MCRdata:
    def __init__(self):
        self.graph = list()
        self.ob_label = list()
        self.traj = list()
        self.aabb = list()
        self.circle = list()
        self.radius = list()
        self.planning_time = list()
        self.sectors = list()

class GraphSaveClass:
    def __init__(self,x=list(),edge=list(),y=list()):
        self.x = x
        self.edge = edge
        self.y = y

class MyGraphSaveClass:
    def __init__(self,g,key_configuration_size,y):
        self.g = g
        self.key_configuration_size = key_configuration_size
        self.y = y

