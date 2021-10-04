import torch
import torch.nn as nn
import torch.nn.functional as F

# node update function pi
class Node_Update_Function(nn.Module):
    def __init__(self, in_feature_num, out_feature_num, hidden_layers=(128, 256, 128)):
        super(Node_Update_Function, self).__init__()
        
        self.in_feature_num = in_feature_num
        self.out_feature_num = out_feature_num
        
        self.hidden_layers = hidden_layers
        self.n_hidden_layers = len(hidden_layers)
        
        self.update_function = nn.ModuleList([nn.Linear(in_feature_num, self.hidden_layers[i]) if i == 0 else
                                  nn.Linear(self.hidden_layers[i-1], out_feature_num) if i == self.n_hidden_layers else
                                  nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]) for i in range(self.n_hidden_layers+1)])

        self.relu = torch.relu

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = self.relu(self.update_function[i](x))
        return self.update_function[-1](x)
    
# edge to node aggregation function rho
class Edge_to_Node_Aggregation_Function(nn.Module):
    def __init__(self, in_feature_num, mid_feature_num, out_feature_num,
                    in_mid_hidden_layers=(128, 256, 128), mid_out_hidden_layers=(128, 256, 128)):
        super(Edge_to_Node_Aggregation_Function, self).__init__()
        
        self.in_feature_num = in_feature_num
        self.mid_feature_num = mid_feature_num
        self.out_feature_num = out_feature_num
        
        self.in_mid_hidden_layers = in_mid_hidden_layers
        self.mid_out_hidden_layers = mid_out_hidden_layers

        self.n_in_mid_hidden_layers = len(in_mid_hidden_layers)
        self.n_mid_out_hidden_layers = len(mid_out_hidden_layers)

        # in_mid aggregation layer
        self.in_mid_aggregation_layers = nn.ModuleList([nn.Linear(self.in_feature_num, self.in_mid_hidden_layers[i]) if i == 0 else
                                            nn.Linear(self.in_mid_hidden_layers[i-1], self.mid_feature_num) if i == self.n_in_mid_hidden_layers else
                                            nn.Linear(self.in_mid_hidden_layers[i-1], self.in_mid_hidden_layers[i]) for i in range(self.n_in_mid_hidden_layers+1)])

        # mid_out aggregation layer
        self.mid_out_aggregation_layer = nn.ModuleList([nn.Linear(self.mid_feature_num, self.mid_out_hidden_layers[i]) if i == 0 else
                                            nn.Linear(self.mid_out_hidden_layers[i-1], self.out_feature_num) if i == self.n_mid_out_hidden_layers else
                                            nn.Linear(self.mid_out_hidden_layers[i-1], self.mid_out_hidden_layers[i]) for i in range(self.n_mid_out_hidden_layers+1)])

        self.dropout = nn.Dropout(0.2)

        self.relu = torch.relu

    def in_mid_aggregation_function(self, x):
        x = x[0]
        for i in range(self.n_in_mid_hidden_layers):
            # x = self.relu(self.in_mid_aggregation_layers[i](x))
            x = self.dropout(self.relu(self.in_mid_aggregation_layers[i](x)))
        return self.in_mid_aggregation_layers[-1](x)

    def mid_out_aggregation_function(self, x, adjancy_matrix):
        x = x.unsqueeze(0)
        h_x = torch.bmm(adjancy_matrix, x)
        s = adjancy_matrix.sum(dim=1)
        if len(s)==1:
            s = s.view(-1,1)
        else:
            s = s.view(-1,len(s),1)
        h_x = torch.div(h_x, s)

        for i in range(self.n_mid_out_hidden_layers):
            # h_x = self.relu(self.mid_out_aggregation_layer[i](h_x))
            h_x = self.dropout(self.relu(self.mid_out_aggregation_layer[i](h_x)))
        return self.mid_out_aggregation_layer[-1](h_x)

    def forward(self, x, adjancy_matrix):
        return self.mid_out_aggregation_function(self.in_mid_aggregation_function(x), adjancy_matrix)

class GNN_clf(nn.Module):
    def __init__(self, node_in_feature_num, node_out_feature_num,
                    edge_in_feature_num, edge_mid_feature_num, edge_out_feature_num,
                    task="classification",
                    node_hidden_layers=(128, 256, 128),
                    edge_in_mid_hidden_layers=(128, 256, 128), edge_mid_out_hidden_layers=(128, 256, 128),
                    cuda=False):
        super(GNN_clf, self).__init__()
        self.aggregation_layer = Edge_to_Node_Aggregation_Function(edge_in_feature_num, edge_mid_feature_num, edge_out_feature_num, edge_in_mid_hidden_layers, edge_mid_out_hidden_layers)
        self.update_layer = Node_Update_Function(node_in_feature_num+edge_out_feature_num, node_out_feature_num, node_hidden_layers)
        if cuda:
            self.aggregation_layer.cuda()
            self.update_layer.cuda()
        self.task = task
        if self.task=="classification":
            self.classifier = nn.Linear(node_out_feature_num, 1)
            self.sig = torch.sigmoid

    def forward(self, x, adjancy_matrix):
        h_x = self.aggregation_layer(x, adjancy_matrix)
        h_x = torch.cat([x, h_x], dim=2)
        h_x = self.update_layer(h_x)
        if self.task=="classification":
            h_x = self.classifier(h_x)
            return self.sig(h_x)
        else:
            return h_x


# test
if __name__=='__main__':

    pnet = GNN_clf(2,100,2,3,4)
    print(pnet)
    net = Node_Update_Function(100,100)
    print(net)
    net2 = Edge_to_Node_Aggregation_Function(100,200,3)
    print(net2)
        
