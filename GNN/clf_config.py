from typing import List, Tuple, Dict
model_config : Dict = dict()
model_config['epochs'] = 500
model_config['learning_rate'] = 1e-4
model_config['CUDA'] = True
model_config['log_step'] = 1
model_config['save_step'] = 30
model_config['TRAIN'] = True
model_config['plot_mAP'] = False
model_config['probability_threshold'] = 0.5
model_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
model_config['batch_size'] = 32
