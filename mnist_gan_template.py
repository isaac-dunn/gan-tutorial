import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import IterableDataset, DataLoader

torch.manual_seed(0)

def weights_init():
    def fn_to_return(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal(module.weight)
    return fn_to_return

class Generator(nn.Module):
    pass # TODO

class Discriminator(nn.Module):
    pass # TODO

# note that this will download MNIST for you: just point the path to an empty directory
mnist_dataset = torchvision.datasets.MNIST('data', download=True, transform=torchvision.transforms.ToTensor())
mnist_dataloader = DataLoader(mnist_dataset, batch_size=100)

g = Generator()
d = Discriminator()

g_opt = torch.optim.Adam(g.parameters(), lr=0.00001)
d_opt = torch.optim.Adam(d.parameters(), lr=0.00001)

iteration = 0
while True:
    for batch, label in mnist_dataloader:
        iteration += 1
        batch_size = batch.shape[0]

        # TODO: train GAN here: can be copied from work so far
        
        if iteration % 100 == 0:
            torchvision.utils.save_image(batch, 'real.png', normalize=True, range=(0,1))
            torchvision.utils.save_image(gen_batch, 'fake.png', normalize=True, range=(0,1))


