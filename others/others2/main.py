
from typing import List, Dict, Tuple, Set
import random

from matplotlib.pyplot import step
from Graph import Graph

from MCRPlanner import MCRPlanner
from Utils import draw_initial_map
from World import obstacle, World



def generate_obstacles(num, radius, width, height):
    obstacles = list()
    modes = [[],[],[]]

    # rectangle static obstacles
    obstacles.append(obstacle(shape='rectangle', is_static=True, index=0, half_width=4, half_height=1.5, center_x=6, center_y=3.5))
    obstacles.append(obstacle(shape='rectangle', is_static=True, index=1, half_width=4, half_height=1.5, center_x=6, center_y=8.5))

    for idx in range(2,2+num):
        dice = random.random()
        if dice < 0.33:
            coord_x = 2 + random.random() * 8
            coord_y = random.random() * 2
            modes[0].append(idx)
            obstacles.append(obstacle(shape='circle', is_static=False, index=idx, pos=(coord_x, coord_y), radius=radius))

        elif dice < 0.66:
            coord_x = 2 + random.random() * 8
            coord_y = 5 + random.random() * 2
            modes[1].append(idx)
            obstacles.append(obstacle(shape='circle', is_static=False, index=idx, pos=(coord_x, coord_y), radius=radius))

        else:
            coord_x = 2 + random.random() * 8
            coord_y = 10 + random.random() * 2
            modes[2].append(idx)
            obstacles.append(obstacle(shape='circle', is_static=False, index=idx, pos=(coord_x, coord_y), radius=radius))
    return obstacles, modes


if __name__=='__main__':

    # generate obstacles
    obstacles, modes = generate_obstacles(num=1, radius=1.0, width=12.0, height=12.0)

    # generate start and goal configuration
    q_s = [random.random() * 2, random.random() * 2]
    q_g = [10 + random.random() * 2, 10 + random.random() * 2]

    # generate world
    world = World(q_s=q_s, q_g=q_g, world_size=[12.0,12.0], dim_config=2, step=0.1, obstacles=obstacles)

    # draw initial map
    draw_initial_map(world)

    # generate graph
    graph = Graph(q_s=q_s, q_g=q_g)

    planner = MCRPlanner(world=world, graph=graph, delta=0.5, m=10, N_raise=1000)

    planner.continuousMCR(500)
    print(planner.traj)

    


    