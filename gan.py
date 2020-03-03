import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np

matplotlib.use('tkagg')
plt.ion()
fig, ax = plt.subplots()


class GaussiansIterable:
    def __init__(self):
        self.gaussians = [
            #((3, 1), 0.1),
            #((2, 6),0.11),
            ((7, 7),1),
            ((3, 3),1),
        ]
    def __iter__(self):
        return self
    def __next__(self):
        (centre, std) = random.choice(self.gaussians)
        return np.random.normal(centre, std).astype(float)

class GaussiansDataset(IterableDataset):
    def __iter__(self):
        return GaussiansIterable()

def plot(real, generated):
    plt.cla()
    plt.scatter(real[:,0], real[:,1], c='orange')
    plt.scatter(generated[:,0], generated[:,1], c='purple')
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    fig.canvas.draw_idle()
    plt.pause(0.0001)

class Generator(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
                nn.Linear(latent_dim, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 20),
                nn.LeakyReLU(0.2),
                nn.Linear(20, 20),
                nn.LeakyReLU(0.2),
                nn.Linear(20, 20),
                nn.LeakyReLU(0.2),
                nn.Linear(20, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 2),
                nn.Sigmoid()
            )
    def forward(self, z):
        return 10*self.layers(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(2, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 1),
                #nn.Sigmoid()
            )
    def forward(self, point):
        return self.layers(point)

dataloader = DataLoader(GaussiansDataset(), batch_size=100)

g = Generator()
d = Discriminator()

g_opt = torch.optim.Adam(g.parameters(), lr=0.001)
d_opt = torch.optim.Adam(d.parameters(), lr=0.001)

for batch in dataloader:
    batch = batch.float()
    batch_size = batch.shape[0]
    gen_batch = g(torch.randn(batch_size, g.latent_dim))

    # train d: minimise -log(d(x)) -log(1-d(g(z)))
    d_opt.zero_grad()
    #loss_d = F.binary_cross_entropy(d(batch), torch.ones(batch_size,1)) +\
    #       F.binary_cross_entropy(d(gen_batch.detach()), torch.zeros(batch_size,1))
    loss_d = torch.mean(d(batch)) - torch.mean(d(gen_batch))
    loss_d.backward()
    d_opt.step()

    # train g: minimise +log(1-d(g(z)))
    g_opt.zero_grad()
    loss_g = -torch.mean(d(gen_batch))
    #loss_g = -F.binary_cross_entropy(d(gen_batch), torch.zeros(batch_size,1))
    #loss_g = F.binary_cross_entropy(d(gen_batch), torch.ones(batch_size,1))
    loss_g.backward()
    g_opt.step()
    
    plot(batch, gen_batch.detach())

