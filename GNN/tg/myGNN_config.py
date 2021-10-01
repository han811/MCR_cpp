from typing import List, Tuple, Dict

# GCN model parameters
GCN_config : Dict = dict()
GCN_config['hidden_channels'] = 2048
GCN_config['dropout'] = 0.2
GCN_config['activation'] = 'elu'

# SAGE model parameters
SAGE_config : Dict = dict()
SAGE_config['hidden_channels'] = 8
SAGE_config['dropout'] = 0.2
SAGE_config['activation'] = 'elu'

# GAT model parameters
GAT_config : Dict = dict()
GAT_config['hidden_channels'] = 2
GAT_config['dropout'] = 0.4
GAT_config['activation'] = 'elu'

# GATcVAE model parameters
GATcVAE_config : Dict = dict()
GATcVAE_config['en_hidden_channels'] = 512
GATcVAE_config['de_hidden_channels'] = 512
GATcVAE_config['activation'] = 'elu'
GATcVAE_config['dropout'] = 0.2
GATcVAE_config['z_dim'] = 64

