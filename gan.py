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


class GaussiansIterable:
    def __init__(self):
        self.gaussians = [
            #((3, 1), 0.1),
            #((2, 6),0.11),
            ((7, 7),1),
            ((3, 3),0.4),
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

def weights_init():
    def fn_to_return(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal(module.weight)
    return fn_to_return

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
        self.apply(weights_init())
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
                nn.Sigmoid()
            )
    def forward(self, point):
        return self.layers(point)

def compute_gradient_penalty(batch, gen_batch, d):
    batch_size = batch.shape[0]
    alpha = torch.rand(batch_size,1)
    interpolates = alpha*batch + (1-alpha)*gen_batch
    gradients = torch.autograd.grad(
            outputs=d(interpolates),
            inputs=interpolates,
            grad_outputs=torch.ones(batch_size, 1),
        )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

dataloader = DataLoader(GaussiansDataset(), batch_size=100)

g = Generator()
d = Discriminator()

g_opt = torch.optim.Adam(g.parameters(), lr=0.00001)
d_opt = torch.optim.Adam(d.parameters(), lr=0.00001)

iteration = 0
for batch in dataloader:
    iteration += 1
    batch = batch.float()
    batch_size = batch.shape[0]
    gen_batch = g(torch.randn(batch_size, g.latent_dim))

    # train d: minimise -log(d(x)) -log(1-d(g(z)))
    d_opt.zero_grad()
    loss_d = F.binary_cross_entropy(d(batch), torch.ones(batch_size,1)) +\
           F.binary_cross_entropy(d(gen_batch.detach()), torch.zeros(batch_size,1))
    #loss_d = torch.mean(d(batch)) - torch.mean(d(gen_batch.detach())) + compute_gradient_penalty(batch, gen_batch, d)
    loss_d.backward()
    d_opt.step()

    # train g: minimise +log(1-d(g(z)))
    g_opt.zero_grad()
    #loss_g = torch.mean(d(gen_batch))
    loss_g = -F.binary_cross_entropy(d(gen_batch), torch.zeros(batch_size,1))
    #loss_g = F.binary_cross_entropy(d(gen_batch), torch.ones(batch_size,1))
    print(loss_g.item())
    loss_g.backward()
    g_opt.step()
    
    if iteration % 100 == 0:
        plot(batch, gen_batch.detach())


