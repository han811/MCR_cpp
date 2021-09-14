import matplotlib.pyplot as plt

def plot_with_labels(label):
# def plot_with_labels():
    start_point = plt.Circle((-20, 0), 0.5, color='r')
    goal_point = plt.Circle((20, 0), 0.5, color='b')

    fig, ax = plt.subplots()

    plt.xlim(-30,30)
    plt.ylim(-30,30)

    # plt.grid(linestyle='--')

    ax.add_artist(start_point)
    ax.add_artist(goal_point)

    plt.title('Plot validation', fontsize=10)

    # plt.savefig("plot_circle_matplotlib_02.png", bbox_inches='tight')

    for ob_x, ob_y in label[2]:
        obstacle= plt.Circle((ob_x, ob_y), 1.5, color='black', alpha=0.2)
        ax.add_artist(obstacle)
        
    
    
    plt.show()
    
if __name__=="__main__":
    print("plot_with_labels")