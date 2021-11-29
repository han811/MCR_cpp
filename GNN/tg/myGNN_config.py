from typing import Dict

# GCN model parameters
GCN_config : Dict = dict()
GCN_config['hidden_channels'] = 32
GCN_config['dropout'] = 0.2
GCN_config['activation'] = 'elu'

# SAGE model parameters
SAGE_config : Dict = dict()
<<<<<<< HEAD
SAGE_config['hidden_channels'] = 2
SAGE_config['num_layers'] = 1
=======
SAGE_config['hidden_channels'] = 16
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
SAGE_config['dropout'] = 0.2
SAGE_config['activation'] = 'elu'

# GAT model parameters
GAT_config : Dict = dict()
GAT_config['hidden_channels'] = 8
GAT_config['dropout'] = 0.2
GAT_config['activation'] = 'elu'

# GATcVAE model parameters
GATcVAE_config : Dict = dict()
GATcVAE_config['en_hidden_channels'] = 32
GATcVAE_config['de_hidden_channels'] = 32
# GATcVAE_config['en_hidden_channels'] = 512
# GATcVAE_config['de_hidden_channels'] = 512
GATcVAE_config['activation'] = 'elu'
GATcVAE_config['dropout'] = 0.2
GATcVAE_config['z_dim'] = 64

# subgraph AE model parameters
GAE_config : Dict = dict()
GAE_config['in_channels'] = 2
GAE_config['hidden_channels'] = 32
GAE_config['latent_channels'] = 8
GAE_config['activation'] = 'elu'
GAE_config['save_name'] = 'my_model_AE_2021-10-05_13-52.pt'


# GNN-cVAE model parameters
SAGEcVAE_config: Dict = dict()
<<<<<<< HEAD
SAGEcVAE_config['embedding_hidden_channels'] = 16
SAGEcVAE_config['embedding_channels'] = 32
SAGEcVAE_config['en_hidden_channels'] = 128
SAGEcVAE_config['de_hidden_channels'] = 128
SAGEcVAE_config['z_dim'] = 32
SAGEcVAE_config['activation'] = 'elu'
SAGEcVAE_config['dropout'] = 0.25
=======
SAGEcVAE_config['embedding_hidden_channels'] = 64
SAGEcVAE_config['embedding_channels'] = 32
SAGEcVAE_config['en_hidden_channels'] = 16
SAGEcVAE_config['de_hidden_channels'] = 16
SAGEcVAE_config['z_dim'] = 8
SAGEcVAE_config['activation'] = 'elu'
SAGEcVAE_config['dropout'] = 0.2
>>>>>>> 015381326fbee3f6f117c4a3668e6cdb1e19924e
SAGEcVAE_config['is_save_hiddens'] = False
