from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

class Feature_net(nn.Module):
    def __init__(self, in_f, out_f, hidden_layers=(128, 256, 128), activation=None):
        super(Feature_net, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.hidden_layers = hidden_layers
        self.n_hidden_layers = len(hidden_layers)
        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        else:
            self.activation = None
        self.net = nn.ModuleList([nn.Linear(self.in_f, self.hidden_layers[i]) if i == 0 else
                            nn.Linear(self.hidden_layers[i-1], self.out_f) if i == self.n_hidden_layers else
                            nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]) for i in range(self.n_hidden_layers+1)])
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            if self.activation:
                x = self.activation(self.net[i](x))
        return self.net[-1](x)

class VAE_Decoder(nn.Module):
    def __init__(self, in_f, mid_f, dim_z=16,
     message_passing_step=3,
     hidden_layers1=(128, 128),
     hidden_layers2=(128, 128),
     hidden_layers3=(128, 128),
     hidden_layers4=(128, 128),
    #  hidden_layers1=(128, 256, 128),
    #  hidden_layers2=(128, 256, 128),
    #  hidden_layers3=(128, 256, 128),
    #  hidden_layers4=(128, 256, 128),
     activation=None):
        super(VAE_Decoder, self).__init__()
        self.in_f = in_f
        self.mid_f = mid_f
        self.dim_z = dim_z
        self.hidden_layers1 = hidden_layers1
        self.hidden_layers2 = hidden_layers2
        self.hidden_layers3 = hidden_layers3
        self.hidden_layers4 = hidden_layers4

        self.k = message_passing_step
        self.current_update = Feature_net(in_f,mid_f,hidden_layers1,activation='relu')
        self.in_update = Feature_net(in_f,mid_f,hidden_layers2,activation='relu')
        self.fin_update = Feature_net(mid_f,in_f,hidden_layers3,activation='relu')
        self.net = nn.ModuleList([nn.Linear(self.in_f+self.dim_z, self.hidden_layers4[i]) if i == 0 else
                nn.Linear(self.hidden_layers4[i-1], 1) if i == len(self.hidden_layers4) else
                nn.Linear(self.hidden_layers4[i-1], self.hidden_layers4[i]) for i in range(len(self.hidden_layers4)+1)])
        
        self.current_update.apply(init_weights)
        self.in_update.apply(init_weights)
        self.fin_update.apply(init_weights)
        self.net.apply(init_weights)

        self.sigmoid = nn.Sigmoid()
        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
    
    def forward(self, vertex, adjancy_matrix, z):
        z = z.unsqueeze(0)
        x = list()
        for i in vertex:
            # x.append(i.squeeze().unsqueeze(0).clone().detach().requires_grad_(True))
            x.append(i.squeeze().unsqueeze(0).clone().detach().requires_grad_(True).cuda())
        N = len(vertex)
        for _ in range(self.k):
            x2 = [0 for _ in range(N)]
            for i in range(N):
                new_x = [0 for _ in range(N)]
                new_x[i] = self.current_update(x[i])
                for j in range(N):
                    if i!=j:
                        new_x[j] = self.in_update(x[j])
                avg_x = new_x[0]
                for k in range(1,N):
                    avg_x += new_x[k]
                avg_x /= N
                x2[i] = self.fin_update(avg_x)
            x = list(x2)
        y = list()
        for tmp_x in x:
            tmp_x = torch.cat([tmp_x,z],1)
            for i in range(len(self.hidden_layers4)):
                tmp_x = self.activation(self.net[i](tmp_x))
                tmp_x = self.dropout(tmp_x)
            y.append(self.sigmoid(self.net[-1](tmp_x)))
        
        for idx,i in enumerate(y):
            if idx==0:
                return_y = i.squeeze(0)
            else:
                return_y = torch.cat([return_y,i.squeeze(0)],0)
        return return_y

                                
class VAE_Encoder(nn.Module):
    def __init__(self, in_f, mid_f, dim_z=32,
     message_passing_step=3,
     hidden_layers1=(128, 256, 128),
     hidden_layers2=(128, 256, 128),
     hidden_layers3=(128, 256, 128),
     hidden_layers4=(128, 256, 128),
     activation=None):
        super(VAE_Encoder, self).__init__()
        self.in_f = in_f+1
        self.mid_f = mid_f
        self.dim_mu = dim_z
        self.dim_logvar = dim_z
        self.hidden_layers1 = hidden_layers1
        self.hidden_layers2 = hidden_layers2
        self.hidden_layers3 = hidden_layers3
        self.hidden_layers4 = hidden_layers4

        self.k = message_passing_step
        self.current_update = Feature_net(self.in_f,mid_f,hidden_layers1,activation='relu')
        self.in_update = Feature_net(self.in_f,mid_f,hidden_layers2,activation='relu')
        self.fin_update = Feature_net(mid_f,self.in_f,hidden_layers3,activation='relu')

        self.net = nn.ModuleList([nn.Linear(self.in_f, self.hidden_layers4[i]) if i == 0 else
                nn.Linear(self.hidden_layers4[i-1], self.dim_mu+self.dim_logvar) if i == len(self.hidden_layers4) else
                nn.Linear(self.hidden_layers4[i-1], self.hidden_layers4[i]) for i in range(len(self.hidden_layers4)+1)])
        
        self.current_update.apply(init_weights)
        self.in_update.apply(init_weights)
        self.fin_update.apply(init_weights)
        self.net.apply(init_weights)
        
        
        self.sigmoid = nn.Sigmoid()
        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='elu':
            self.activation = nn.ELU()
    
    def forward(self, vertex, adjancy_matrix, label):
        x = list()
        for idx,i in enumerate(vertex):
            # x.append(torch.cat([i.squeeze().unsqueeze(0),label[0][idx].unsqueeze(0).unsqueeze(0)],1))
            # x.append(torch.cat([i.squeeze().unsqueeze(0),label.squeeze()[idx].unsqueeze(0).unsqueeze(0)],1))
            x.append(torch.cat([i.squeeze().unsqueeze(0),label.squeeze()[idx].unsqueeze(0).unsqueeze(0)],1).cuda())
        N = len(vertex)
        for _ in range(self.k):
            x2 = [0 for _ in range(N)]
            for i in range(N):
                new_x = [0 for _ in range(N)]
                new_x[i] = self.current_update(x[i])
                for j in range(N):
                    if i!=j:
                        new_x[j] = self.in_update(x[j])
                avg_x = new_x[0]
                for k in range(1,N):
                    avg_x += new_x[k]
                avg_x /= N
                x2[i] = self.fin_update(avg_x)
            x = list(x2)
            avg_x = x[0]
            for i in range(1,N):
                avg_x += x[i]
            avg_x /= N
        for i in range(len(self.hidden_layers4)):
            avg_x = self.activation(self.net[i](avg_x))
        return self.net[-1](avg_x)





def reparameterize(mu, logvar, dim_z=16, is_train=True):
    if is_train:
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu

def loss_function(recon_x, x, mu, logvar, cuda=True):
    if cuda:
        reconstruction_function = nn.MSELoss().cuda()
    else:
        reconstruction_function = nn.MSELoss()
    mse = reconstruction_function(recon_x.cuda(),x.cuda())
    kld_elements = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_elements).mul_(-0.5)
    return mse, kld

# test
if __name__=='__main__':
    g = Graph(True)
    x = list()
    x.append([13,-2])
    x.append([2,44])
    y = [0,1]
    edge = [[0,1],[1,0]]
    g.add_graph(x, edge, y)
    X, A, y = g[0]
    print(X)
    print(A)
    print(y)
    print("-----------------------------")
    dim_z = 8
    z = torch.randn(dim_z)
    encoder = VAE_Encoder(2,3,activation='relu',dim_z=dim_z)
    decoder = VAE_Decoder(2,3,activation='relu',dim_z=dim_z)
    encoder.cuda()
    decoder.cuda()
    z = z.cuda()

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, betas=(0.5, 0.999))
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, betas=(0.5, 0.999))

    reconstruction_function = nn.MSELoss().cuda()

    for i in range(100):
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        mu_logvar = encoder(X,A,y)
        mu = mu_logvar[0][:dim_z]
        logvar = mu_logvar[0][dim_z:]

        sample_z = reparameterize(mu,logvar,dim_z)

        recon_y = decoder(X,A,sample_z)

        recon_loss, kld_loss = loss_function(recon_y,y,mu,logvar)
        loss_value = recon_loss+kld_loss
        loss_value.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

        print('recon_loss',recon_loss)
        print('kld_loss',kld_loss)
        print('loss_value',loss_value)
        print(recon_y)
        print()
