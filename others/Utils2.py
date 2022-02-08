import pickle

def save_data(file_path, obstacle_num, obstacle_centers, obstacle_displacements, path_configurations, start_point, goal_point):
    save_data = [obstacle_num, obstacle_centers, obstacle_displacements, path_configurations, start_point, goal_point]
    with open(file_path, "wb") as file:
        pickle.dump(save_data, file)
        
def load_data(file_path):
    with open(file_path, "rb") as file:
        load_data = pickle.load(file)
        return load_data