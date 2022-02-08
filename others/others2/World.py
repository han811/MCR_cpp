from typing import List, Dict, Tuple, Set
import math

from Box2D import b2CircleShape, b2PolygonShape
from Box2D import b2Transform

class obstacle:
    
    def __init__(self, shape: str, is_static: bool, index: int,
      pos: tuple = (0,0), radius: float = 0.0, # circle
      half_width: float = 0.0, half_height: float = 0.0, center_x: float = 0.0, center_y: float = 0.0, angle: float = 0.0, # rectangle
      vertices: list = [] # polygon
      ):
        
        self.shape = shape

        # shape information
        if shape == 'circle':
            self.ob = b2CircleShape(pos = pos, radius = radius)
        elif shape == 'rectangle':
            self.ob = b2PolygonShape(box = (half_width, half_height, (center_x, center_y), angle))
        elif shape == 'polygon':
            self.ob = b2PolygonShape(vertices = vertices)
        else:
            print(f'no shape information !!')
            exit()
        
        # static information
        self.is_static = is_static

        # index of obstacle
        self.index = index

    def pointCollisionCheck(self, point: list, transform: b2Transform):
        
        return self.ob.TestPoint(transform, point)

    def segmentCollisionCheck(self, q1: list, q2: list, transform: b2Transform, step: float):

        q_direction = []
        for (i,j) in list(zip(q1,q2)):
            q_direction.append(j-i)
        l2_norm = 0
        for i in q_direction:
            l2_norm += i * i
        l2_norm = math.sqrt(l2_norm)
        for idx in range(len(q_direction)):
            q_direction[idx] /= l2_norm

        n_step = 0
        
        while l2_norm > (n_step * step):
            q_new = []
            for (i,j) in list(zip(q1,q_direction)):
                q_new.append(i + n_step * step * j)
            if self.pointCollisionCheck(q_new, transform):
                return True
            n_step += 1
        return False



class World:

    def __init__(self, q_s: list, q_g: list, world_size: list, dim_config: int, step: float,
      obstacles: list):
        
        # initial and goal node
        self.q_s: list = q_s
        self.q_g: list = q_g

        # size of the world
        self.world_size: list = world_size # 0: width, 1: height
        
        # dimension of configuration space
        self.dim_config: int = dim_config

        # obstacles and instance for collision check
        self.obstacles: list = obstacles
        self.transform = b2Transform()
        self.transform.SetIdentity()
        self.step = step

    def checkPointCover(self, q: list, indexes: list):

        cover: set = set()

        for index in indexes:
            if self.obstacles[index].pointCollisionCheck(q, self.transform):
                cover.add(index)
        
        return cover

    def checkLineCover(self, q1: list, q2: list, indexes: list):

        cover: set = set()

        for index in indexes:
            if self.obstacles[index].segmentCollisionCheck(q1, q2, self.transform, self.step):
                cover.add(index)
            
        return cover
