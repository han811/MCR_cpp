import pickle
import numpy as np
import glob
import os
from tqdm import tqdm



file_list = glob.glob(os.getcwd()+'/data_original/*')
data_num = 0
for file_name in tqdm(file_list):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    n_data = len(data.graph)

    for i in tqdm(range(n_data), leave=False):
        g = []
        for j in range(len(data.graph[i]['V'])):
            g.append(data.graph[i]['V'][j])
        np.save(f'data_npy/graph_node/graph_node{data_num+i+1}',np.array(g,dtype=np.float16))
        np.save(f'data_npy/graph_label/graph_label{data_num+i+1}',np.array(data.ob_label[i],dtype=np.int8))
        g = []
        for j in range(len(data.circle[i])):
            tmp = data.circle[i][j] + [data.radius[i]] 
            g.append(tmp)
        np.save(f'data_npy/graph_circle/graph_circle{data_num+i+1}',np.array(g,dtype=np.float16))
        np.save(f'data_npy/graph_traj/graph_traj{data_num+i+1}',np.array(data.traj[i],dtype=np.int8))
    data_num = data_num + n_data
