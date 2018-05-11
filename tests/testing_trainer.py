import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np

class Trainer:
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, gan, lr=5e-5):
        self.gan = gan
        self.disc_optim = optim.RMSprop(gan.discriminator.parameters(), lr=lr)
        self.gen_optim = optim.RMSprop(gan.generator.parameters(), lr=lr)
        self.avg_real = 0
        self.avg_fake = 0

    def train(self, reals, fakes, clip_coef=.01, n_critic=5):

        # Discrim Updating
        self.disc_optim.zero_grad()
        rs = Variable(reals)
        fs = Variable(fakes)
        r_loss = self.gan.discriminate(rs).mean()
        f_loss = self.gan.discriminate(fs).mean()
        disc_loss = f_loss - r_loss # Seek to maximize r_loss
        disc_loss.backward()
        self.disc_optim.step()

        self.gan.clip_ds_ps(clip_coef) # Wasserstein clipping maneuver
        print("Real:", r_loss.data[0])
        print("Fake:", f_loss.data[0])

