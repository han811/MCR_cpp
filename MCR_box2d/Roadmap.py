from typing import List


class Roadmap:
    
    def __init__(self, q):
        self.q = q
        self.parent = None
        self.children : List = list()

    def setParent(self, parent):
        self.parent = parent
    
    def setChild(self, child):
        self.children.append(child)
    
    def getParent(self):
        return self.parent
    
    def getChildren(self):
        return self.children

    