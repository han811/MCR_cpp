from typing import List, Tuple, Dict

# basic GNN model
model_config : Dict = dict()
model_config['epochs'] = 256
model_config['learning_rate'] = 1e-4
model_config['CUDA'] = True
model_config['log_step'] = 1
model_config['save_step'] = 30
model_config['TRAIN'] = True
model_config['plot_mAP'] = False
model_config['probability_threshold'] = 0.5
model_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
model_config['PREPARE'] = True


# basic GNN model test
model_test_config : Dict = dict()
model_test_config['epochs'] = 256
model_test_config['learning_rate'] = 1e-3
model_test_config['CUDA'] = True
model_test_config['log_step'] = 1
model_test_config['save_step'] = 30
model_test_config['TRAIN'] = True
model_test_config['plot_mAP'] = False
model_test_config['probability_threshold'] = 0.5
model_test_config['model_path'] = "./save_model/2021-09-22_14-45.pt"
model_test_config['PREPARE'] = True


# GNN clf model
model_clf_config : Dict = dict()
model_clf_config['epochs'] = 320
model_clf_config['learning_rate'] = 1e-4
model_clf_config['CUDA'] = True
model_clf_config['log_step'] = 1
model_clf_config['save_step'] = 30
model_clf_config['TRAIN'] = True
model_clf_config['plot_mAP'] = False
model_clf_config['probability_threshold'] = 0.5
model_clf_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
model_clf_config['batch_size'] = 32
model_clf_config['n_encoding_feature'] = 32
model_clf_config['n_mid_feature'] = 64


# GNN clf model torch geometric
model_clf_torch_geometric_config : Dict = dict()
model_clf_torch_geometric_config['epochs'] = 320
model_clf_torch_geometric_config['learning_rate'] = 1e-4
model_clf_torch_geometric_config['CUDA'] = True
model_clf_torch_geometric_config['log_step'] = 1
model_clf_torch_geometric_config['save_step'] = 30
model_clf_torch_geometric_config['TRAIN'] = True
model_clf_torch_geometric_config['plot_mAP'] = False
model_clf_torch_geometric_config['probability_threshold'] = 0.5
model_clf_torch_geometric_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
model_clf_torch_geometric_config['batch_size'] = 32
model_clf_torch_geometric_config['n_encoding_feature'] = 32
model_clf_torch_geometric_config['n_mid_feature'] = 64
