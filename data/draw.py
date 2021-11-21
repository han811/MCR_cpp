import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm

from data_class import MCRdata

if __name__=='__main__':
    with open('CollectedData/data_original/MCR_data_2021-11-10_19-11.pickle','rb') as f:
        data = pickle.load(f)

    fig, ax = plt.subplots() 
    fig.set_figheight(80)
    fig.set_figwidth(80)
    ax.set_xlim(0,12.0)
    ax.set_ylim(0,12.0)
    for idx in tqdm(range(len(data.graph[0:10])), desc='draw each graph'):
        ax.clear()
        circles = data.circle[idx]
        radius = data.radius[idx]
        for _,tmp_circle in circles.items():
            circle = plt.Circle((tmp_circle[0],tmp_circle[1]), radius, color='red', alpha=0.15)
            ax.add_patch(circle)
        graph = data.graph[idx]
        for i in list(graph['V'].keys()):
            if i==0:
                circle = plt.Circle((graph['V'][i][0],graph['V'][i][1]), 0.05, color='green', alpha=1)
                ax.add_patch(circle)
            elif i==1:
                circle = plt.Circle((graph['V'][i][0],graph['V'][i][1]), 0.05, color='green', alpha=1)
                ax.add_patch(circle)
            else:
                circle = plt.Circle((graph['V'][i][0],graph['V'][i][1]), 0.025, color='blue', alpha=0.1)
                ax.add_patch(circle)
        for i in list(graph['E'].keys()):
            for e in graph['E'][i]:
                ax.plot([graph['V'][i][0],graph['V'][e][0]],[graph['V'][i][1],graph['V'][e][1]],c='black',linewidth=0.75)
        traj = data.traj[idx]
        for n,i in enumerate(traj):
            if n!=0:
                ax.plot([graph['V'][pre_i][0],graph['V'][i][0]],[graph['V'][pre_i][1],graph['V'][i][1]],c='blue',linewidth=7.5)
            pre_i = i
        ob_label = data.ob_label[idx]
        for ob_index in ob_label:
            circle = plt.Circle((circles[ob_index-1][0],circles[ob_index-1][1]), radius, color='yellow', alpha=0.25)
            ax.add_patch(circle)
        fig.savefig(f'./images/figure{idx+1}_{data.sectors[idx][0]}_{data.sectors[idx][1]}_{data.sectors[idx][2]}_{data.sectors[idx][3]}_{len(ob_label)}_{data.planning_time[idx]}.png')