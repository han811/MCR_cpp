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

class GNN_Generator(nn.Module):
    def __init__(self, in_f, mid_f, dim_z=32,
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
        super(GNN_Generator, self).__init__()
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
            x.append(i.squeeze().unsqueeze(0).clone().detach().requires_grad_(True))
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
        return y

                                
class GNN_Discriminator(nn.Module):
    def __init__(self, in_f, mid_f, dim_z=32,
     message_passing_step=3,
     hidden_layers1=(128, 256, 128),
     hidden_layers2=(128, 256, 128),
     hidden_layers3=(128, 256, 128),
     hidden_layers4=(128, 256, 128),
     activation=None):
        super(GNN_Discriminator, self).__init__()
        self.in_f = in_f+1
        self.mid_f = mid_f
        self.dim_z = dim_z
        self.hidden_layers1 = hidden_layers1
        self.hidden_layers2 = hidden_layers2
        self.hidden_layers3 = hidden_layers3
        self.hidden_layers4 = hidden_layers4

        self.k = message_passing_step
        self.current_update = Feature_net(in_f+1,mid_f,hidden_layers1,activation='relu')
        self.in_update = Feature_net(in_f+1,mid_f,hidden_layers2,activation='relu')
        self.fin_update = Feature_net(mid_f,in_f+1,hidden_layers3,activation='relu')

        self.net = nn.ModuleList([nn.Linear(self.in_f, self.hidden_layers4[i]) if i == 0 else
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
    
    def forward(self, vertex, adjancy_matrix, label):
        x = list()
        for idx,i in enumerate(vertex):
            x.append(torch.cat([i.squeeze().unsqueeze(0),label[0][idx].unsqueeze(0).unsqueeze(0)],1))
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
        return self.sigmoid(self.net[-1](avg_x))






# test
if __name__=='__main__':
    g = Graph(True)
    x = dict()
    x[0] = [13,-2]
    x[1] = [2,44]
    y = [0,1]
    edge = [[0,1],[1,0]]
    g.add_graph(x, edge, y)
    X, A, y = g[0]
    print(X)
    print(A)
    print(y)
    print("-----------------------------")
    z = torch.randn(32)
    generator = GNN_Generator(2,3,activation='relu')
    discriminator = GNN_Discriminator(2,3,activation='relu')
    generator.cuda()
    discriminator.cuda()
    z = z.cuda()

    generator_opt = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    adversarial_criterion = nn.MSELoss().cuda()

    real_label = torch.full((1, 1), 1, dtype=torch.float32).cuda()
    fake_label = torch.full((1, 1), 0, dtype=torch.float32).cuda()

    for i in range(100):
        discriminator.zero_grad()

        real_output = discriminator(X,A,y)
        d_loss_real = adversarial_criterion(real_output, real_label)
        d_loss_real.backward()
        d_x =  real_output.mean()


        noise = torch.randn(32)
        noise = noise.cuda()
        fake = generator(X, A, noise)
        for i in range(len(fake)):
            fake[i] = list(fake[i].cpu().detach().numpy()[0])[0]
        fake = torch.tensor(fake, dtype=torch.float32)
        fake = fake.cuda()
        fake_output = discriminator(X,A,fake)
        d_loss_fake = adversarial_criterion(fake_output, fake_label)
        d_loss_fake.backward()
        d_g_z1 = fake_output.mean()

        d_loss = d_loss_real + d_loss_fake
        discriminator_opt.step()

        generator.zero_grad()

        fake_output = discriminator(X,A,fake)
        g_loss = adversarial_criterion(fake_output, real_label)
        g_loss.backward()
        d_g_z2 = fake_output.mean()
        generator_opt.step()
        
        print('g_loss',g_loss)
        print('d_loss',d_loss)
        print(fake)
        print()
