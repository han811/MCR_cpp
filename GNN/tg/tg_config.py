from typing import List, Tuple, Dict

# GNN clf model torch geometric
tg_clf_config : Dict = dict()
tg_clf_config['epochs'] = 512
tg_clf_config['learning_rate'] = 1e-4
tg_clf_config['CUDA'] = True
tg_clf_config['log_step'] = 1
tg_clf_config['save_step'] = 30
tg_clf_config['TRAIN'] = True
tg_clf_config['plot_mAP'] = False
tg_clf_config['probability_threshold'] = 0.5
tg_clf_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
tg_clf_config['batch_size'] = 32
tg_clf_config['n_encoding_feature'] = 16
tg_clf_config['in_node'] = 1199
# tg_clf_config['in_node'] = 17338
tg_clf_config['tg_clf_config'] = [0,3]

# GNN cVAE model torch geometric
tg_cVAE_config : Dict = dict()
tg_cVAE_config['epochs'] = 512
tg_cVAE_config['learning_rate'] = 1e-4
tg_cVAE_config['CUDA'] = True
tg_cVAE_config['log_step'] = 1
tg_cVAE_config['save_step'] = 30
tg_cVAE_config['TRAIN'] = True
tg_cVAE_config['plot_mAP'] = False
tg_cVAE_config['probability_threshold'] = 0.5
tg_cVAE_config['model_path'] = "./save_model/2021-08-04_15-35.pt"
tg_cVAE_config['batch_size'] = 32
tg_cVAE_config['n_encoding_feature'] = 16
tg_cVAE_config['in_node'] = 1199
tg_cVAE_config['tg_clf_config'] = [0,3]
tg_cVAE_config['beta'] = 0.4
