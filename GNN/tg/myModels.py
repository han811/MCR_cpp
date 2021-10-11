import torch.nn as nn


class mySimpleFC(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super(mySimpleFC, self).__init__()

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, edge):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
