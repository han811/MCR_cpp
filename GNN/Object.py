import math

import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as sg
import shapely.ops as so

class Robot:
    def __init__(self, init_position=(0.0,0.0), robot_type="circle", radius=0.01, polygon_points=None):
        if robot_type=="circle":
            self.robot = sg.Point(init_position[0], init_position[1]).buffer(radius)
            self.robot_radius = radius
        elif robot_type=="polygon":
            self.robot = sg.Polygon(polygon_points)
    
    def feasibility_check(self, geometry):
        return self.robot.intersects(geometry)
    
    def plot(self):
        x,y = self.robot.exterior.xy
        plt.fill(x,y,facecolor='red')

class Obstacle:
    def __init__(self, init_position=(0.0,0.0), obstacle_type="circle", radius=0.01, polygon_points=None, static=False):
        if obstacle_type=="circle":
            self.ob = sg.Point(init_position[0], init_position[1]).buffer(radius) #, resolution=32)
            self.ob_radius = radius
            self.x = init_position[0]
            self.y = init_position[1]
        elif obstacle_type=="polygon":
            self.ob = sg.Polygon(polygon_points)
        self.static = static
    
    def feasibility_check(self, geometry):
        if self.ob.intersects(geometry):
            return False
        else:
            return True
    
    def plot(self, moved=False, check=False):
        x,y = self.ob.exterior.xy
        if not moved:
            if self.static:
                plt.gca().fill(x,y,facecolor='black')
            else:
                plt.gca().fill(x,y,facecolor='gray')
        else:
            if not check:
                plt.gca().fill(x,y,facecolor='gray',alpha=0.4)
            else:
                plt.gca().fill(x,y,facecolor='yellow')

        
class Obstacles:
    def __init__(self, num, obs_radius, static_obstacles, mwidth, mheight, movable_obstacle_type="circle", static_obstacle_type="polygon"):
        self.obs_radius = obs_radius
        self.static_obstacles = []
        self.ob_num = 0
        
        for static_ob in static_obstacles:
            self.static_obstacles.append(Obstacle(obstacle_type="polygon", polygon_points=static_ob, static=True))
            self.ob_num += 1
        self.ob_num += num
        self.movable_ob_num = num
        
        self.mwidth = mwidth
        self.mheight = mheight
        
        self.movable_obstacle_type = movable_obstacle_type
        self.static_obstacle_type = static_obstacle_type
        
        self.movable_obstacles = []
        
        
        for idx in range(num):
            while(True):
                loop_sig = False
                
                tmp_center = (np.random.rand(2) - 0.5)
                tmp_center[0] *= mwidth
                tmp_center[1] *= mheight
                
                tmp_ob = sg.Point(tmp_center).buffer(self.obs_radius)
                
                for obstacle in self.static_obstacles:
                    if not obstacle.feasibility_check(tmp_ob):
                        loop_sig = True
                
                for obstacle in self.movable_obstacles:
                    if not obstacle.feasibility_check(tmp_ob):
                        loop_sig = True
                
                if loop_sig:
                    continue
                else:
                    self.movable_obstacles.append(Obstacle(init_position=tmp_center, radius=self.obs_radius))
                    break;

    def plot(self):
        for obstacle in self.movable_obstacles:
            obstacle.plot()
            
        for obstacle in self.static_obstacles:
            obstacle.plot()
            
    def feasibility_check(self, geometry):
        for obstacle in self.movable_obstacles:
            if not obstacle.feasibility_check(geometry):
                return False
        
        for obstacle in self.static_obstacles:
            if not obstacle.feasibility_check(geometry):
                return False
        
        return True

if __name__=='__main__':
    ob = Obstacles(10, 1, [[[0,0],[1,0],[0,1],[0,0]]], 20, 20)
    ob.plot()
    plt.show()
    # ob = Obstacle(1,20,20,[10,10],3,[[0.0,0.0,1.0,1.0]])
    # ob.plot()
    # Robot().plot_robot()
            
