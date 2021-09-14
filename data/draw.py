import pickle

from matplotlib import pyplot as plt

from data_class import MCRdata

if __name__=='__main__':
    with open('MCR_data.pickle','rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(30,40))
    fig, ax = plt.subplots() 
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((0.0, 5.0))
    ax.set_ylim((0.0, 5.0))

    for idx in range(len(data.graph)):
        plt.cla()
        circles = data.circle[idx]
        radius = data.radius[idx]
        for _,tmp_circle in circles.items():
            plt.gca().scatter(tmp_circle[0],tmp_circle[1],c='red',s=2*radius*2000.0,alpha=0.1)
        graph = data.graph[idx]
        for i in list(graph['V'].keys()):
            if i==0:
                plt.gca().scatter(graph['V'][i][0],graph['V'][i][1],c='green',s=10.0,alpha=1.0)
            elif i==1:
                plt.gca().scatter(graph['V'][i][0],graph['V'][i][1],c='green',s=10.0,alpha=1.0)
            else:
                plt.gca().scatter(graph['V'][i][0],graph['V'][i][1],c='blue',s=0.1,alpha=1.0)
        for i in list(graph['E'].keys()):
            for e in graph['E'][i]:
                plt.gca().plot([graph['V'][i][0],graph['V'][e][0]],[graph['V'][i][1],graph['V'][e][1]],c='black')
        
        traj = data.traj[idx]
        for n,i in enumerate(traj):
            if n!=0:
                plt.gca().plot([graph['V'][pre_i][0],graph['V'][i][0]],[graph['V'][pre_i][1],graph['V'][i][1]],c='blue')
            pre_i = i

        ob_label = data.ob_label[idx]
        for ob_index in ob_label:
            plt.gca().scatter(circles[ob_index-2][0],circles[ob_index-2][1],c='yellow',s=2*radius*2000.0,alpha=0.25)

        plt.xlabel(f"{data.sectors[idx][0]}_{data.sectors[idx][1]}_{data.sectors[idx][2]}_{len(ob_label)}_{data.planning_time[idx]}")

        plt.savefig(f'./images/figure{idx}.png')