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
    def __init__(self,x=list(),edge=list(),y=list(),key_configuration_size=0):
        self.x = x
        self.edge = edge
        self.y = y
        self.key_configuration_size = key_configuration_size