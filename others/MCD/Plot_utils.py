import math

import numpy as np
from matplotlib import pyplot as plt

def plot_world():
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
