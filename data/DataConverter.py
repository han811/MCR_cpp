import pickle

from matplotlib import pyplot as plt
from data_class import MCRdata


if __name__=="__main__":
    data = MCRdata()
    total_data_size = 10
    data_range = range(total_data_size)
    for data_num in data_range:
        try:
            f = open(f"data{data_num}.txt","r")
            Lines = f.readlines()
            graph = dict()
            nodes = dict()
            edges = dict()
            traj = list()
            ob_label = list()
            aabb = dict()
            radius = 0.0
            circle = dict()
            planning_time = 0.0
            
            idx = 0
            idx2 = idx
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='Edges'):
                    break
                node = Lines[idx2].strip().split()
                nodes[idx2-idx-1] = [node[0],node[1]]
            idx = idx2
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='Path'):
                    break
                edge = Lines[idx2].strip().split()
                edges[int(edge[0])] = []
                for i in range(2,len(edge)):
                    edges[int(edge[0])].append(int(edge[i]))
            graph['V'] = nodes
            graph['E'] = edges
            idx = idx2
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='Cover'):
                    break
                traj = Lines[idx2].strip().split()
                traj = list(map(int,traj))
            idx = idx2
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='Obstacles'):
                    break
                ob = Lines[idx2].strip().split()
                for i in range(len(ob)):
                    ob_label.append(int(ob[i][4:-1]))
            idx = idx2+1
            idx2 = idx
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='circles'):
                    break
                aabb_num = int(Lines[idx2].strip()[0])
                idx2+=1
                aabb_bmin = Lines[idx2].strip().split()
                aabb_bmin = list(map(float,aabb_bmin))
                idx2+=1
                aabb_bmax = Lines[idx2].strip().split()
                aabb_bmax = list(map(float,aabb_bmax))
                aabb[aabb_num] = [aabb_bmin,aabb_bmax]
            idx = idx2
            idx2 += 1
            radius = float(Lines[idx2].strip()[0])
            n = 0
            while True:
                idx2+=1
                if(Lines[idx2].strip()=='Time'):
                    break
                c = Lines[idx2].strip().split()
                c = list(map(float,c))
                circle[n] = c
                n+=1
            idx = idx2
            idx += 1
            planning_time = float(Lines[idx].strip().split()[0])


            data.graph.append(graph)
            data.ob_label.append(ob_label)
            data.traj.append(traj)
            data.aabb.append(aabb)
            data.circle.append(circle)
            data.radius.append(radius)
            data.planning_time.append(planning_time)
        except FileNotFoundError as e:
            print(e)
            pass
    with open('MCR_data.pickle','wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # with open('MCR_data.pickle','rb') as f:
    #     read_data = pickle.load(f)
