import time

import Box2D
from Box2D import b2World
from Box2D import b2CircleShape, b2PolygonShape
from Box2D import b2Transform, b2Mat22
from Box2D import b2Draw, b2DrawExtended

circle1 = b2CircleShape(pos=(1,2), radius=0.5)
circle2 = b2CircleShape(pos=(1,2), radius=0.5)
triangle = b2PolygonShape(vertices=[(0,0), (1,0), (0,1)])

'''
example of polygon box
box = b2PolygonShape(box=(half_width, half_height))
box = b2PolygonShape(box=(half_width, half_height, (center_x, center_y), angle))
'''
print(circle1.pos)

transform = b2Transform()
transform.SetIdentity()
print(circle2.TestPoint(transform, (0.48,2)))
print(triangle.TestPoint(transform, (0.5,0.5)))

class World:
    
    def __init__(self, q_s, q_g, width, height, static_obs, dynamic_obs):
        
        # initial and goal configuration
        self.q_s = q_s
        self.q_g = q_g
        self.q_dim = len(q_s)

        # width and height of the world
        self.width = width
        self.height = height

        # obstacles in the world - obstacles are Box2D instances
        self.static_obs = static_obs
        self.dynamic_obs = dynamic_obs

    



    

