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


class myEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, 1024)
        self.linear2 = nn.Linear(1024, out_channels)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
        # return self.sigmoid(self.linear2(self.activation(self.linear1(x))))


class myDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myDecoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, 1024)
        self.linear2 = nn.Linear(1024, out_channels)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.linear(x))
        return self.sigmoid(self.linear2(self.activation(self.linear1(x))))
        # return self.linear2(self.activation(self.linear1(x)))


class myAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(myAutoEncoder, self).__init__()
        self.enc = myEncoder(in_channels, hidden_channels)
        self.dec = myDecoder(hidden_channels, in_channels)

    def forward(self, x):
        return self.dec(self.enc(x))

