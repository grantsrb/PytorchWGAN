import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class GAN(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj
    
    def __init__(self, img_shape, z_size=100, emb_size=200, bnorm=False, z_std=1):
        super(GAN, self).__init__()
        self.emb_size = emb_size
        self.z_size = z_size
        self.img_shape = img_shape
        self.flat_img = int(np.prod(img_shape[-3:]))
        self.bnorm = bnorm
        self.z_std = z_std

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
    
    def generate(self, n, std=1.):
        """
        n - integer denoting number of samples to generate
        std - float or FloatTensor of standard deviations
        """
        zs = Variable(self.cuda_if(torch.randn(n,self.z_size)*self.z_std))
        return self.generator(zs)

    def discriminate(self, x):
        return self.discriminator(x)

    def clip_ds_ps(self, clip_coef):
        for p in self.discriminator.parameters():
            p.data.clamp_(-clip_coef, clip_coef)
