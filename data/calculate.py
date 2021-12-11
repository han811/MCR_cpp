import pickle

from data_class import MCRdata

if __name__=='__main__':
    with open('CollectedData/data_original/MCR_data_2021-12-11_15-53.pickle','rb') as f:
        data = pickle.load(f)
    print("average planning time:",sum(data.planning_time)/len(data.planning_time))
    print("average iteration number:",sum(data.iterations)/len(data.iterations))