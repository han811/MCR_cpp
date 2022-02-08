from matplotlib import pyplot as plt

from World import World, obstacle

def draw_initial_map(world: World):
    
    fig, ax = plt.subplots() 
    fig.set_figheight(80)
    fig.set_figwidth(80)
    ax.set_xlim(0,12.0)
    ax.set_ylim(0,12.0)

    q_s_plot = plt.Circle(world.q_s, 0.05, color='green', alpha=1)
    ax.add_patch(q_s_plot)
    q_g_plot = plt.Circle(world.q_g, 0.05, color='green', alpha=1)
    ax.add_patch(q_g_plot)

    for circle in world.obstacles:
        if circle.shape == 'circle':
            circle_plot = plt.Circle((circle.ob.pos.x,
                circle.ob.pos.y), 
                circle.ob.radius, color='red', alpha=0.15)
            ax.add_patch(circle_plot)

    fig.savefig(f'./images/initial_map.png')
    # ax.clear()