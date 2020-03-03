import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

matplotlib.use('tkagg')
plt.ion()
fig, ax = plt.subplots()

def plot(real, generated):
    plt.cla()
    plt.scatter(real[:,0], real[:,1], c='orange')
    plt.scatter(generated[:,0], generated[:,1], c='purple')
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    fig.canvas.draw_idle()
    plt.pause(0.0001)

class GaussiansIterable:
    def __init__(self):
        self.gaussians = [
            ((7, 7),1),
            #((3, 3),0.4),
        ]
    def __iter__(self):
        return self
    def __next__(self):
        (centre, std) = random.choice(self.gaussians)
        return np.random.normal(centre, std).astype(float)

class GaussiansDataset(IterableDataset):
    def __iter__(self):
        return GaussiansIterable()

def weights_init():
    def fn_to_return(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal(module.weight)
    return fn_to_return

class Generator(nn.Module):
    def __init__(self, latent_dim=10):
        # TODO
    def forward(self, z):
        # TODO

class Discriminator(nn.Module):
    def __init__(self):
        # TODO
    def forward(self, point):
        # TODO

dataloader = DataLoader(GaussiansDataset(), batch_size=100)

g = Generator()
d = Discriminator()

g_opt = torch.optim.Adam(g.parameters(), lr=0.0001)
d_opt = torch.optim.Adam(d.parameters(), lr=0.0001)

iteration = 0
for batch in dataloader:
    iteration += 1
    batch = batch.float()
    batch_size = batch.shape[0]

    gen_batch = # TODO generate batch of fake data

    # train d: minimise -log(d(x)) -log(1-d(g(z)))
    d_opt.zero_grad()
    loss_d = # TODO
    loss_d.backward()
    d_opt.step()

    # train g: minimise +log(1-d(g(z)))
    g_opt.zero_grad()
    loss_g = # TODO
    loss_g.backward()
    g_opt.step()
    
    if iteration % 100 == 0:
        plot(batch, gen_batch.detach())


