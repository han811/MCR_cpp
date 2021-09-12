import pickle

from matplotlib import pyplot as plt

from data_class import MCRdata

if __name__=='__main__':
    with open('MCR_data.pickle','rb') as f:
        data = pickle.load(f)
    for graph in data.graph:
        # for idx in list(graph['V'].keys()):
        #     plt.scatter(graph['V'][idx][0],graph['V'][idx][1])
        # plt.show()
        print(list(graph['V'].keys()))
