import torch.nn
import numpy as np

class GAN(nn.Module()):
    
    def __init__(self, img_shape, z_size=100, emb_size=200, bnorm=False):
        super(GAN, self).__init__()
        self.emb_size = emb_size
        self.z_size = z_size
        self.flat_img = np.prod(img_shape[-3:])
        self.bnorm = bnorm

        # Discriminator
        discrim = []
        discrim.append(nn.Linear(self.flat_img, emb_size))
        discrim.append(nn.ReLU())
        if self.bnorm:
            discrim.append(nn.BatchNorm1d(self.emb_size))
        discrim.append(nn.Linear(self.emb_size, 1))
        self.discriminator = nn.Sequential(*discrim)

        # Generator
        gener = []
        gener.append(nn.Linear(self.z_size, self.emb_size))
        gener.append(nn.ReLU())
        if self.bnorm:
            gener.append(nn.BatchNorm1d(self.emb_size))
        gener.append(nn.Linear(self.emb_size, self.flat_img))
        self.generator = nn.Sequential(*gener)

        def generate(self, z):
            return self.generator(z)

        def discriminate(self, x):
            return self.discriminator(x)
