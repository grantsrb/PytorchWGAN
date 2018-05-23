import torch.nn as nn
from discriminator import Discriminator
from generator import Generator

class DCGAN(nn.Module):

    def __init__(self, img_shape, z_size=100, trainable_z=False, disc_bnorm=True, gen_bnorm=True, vbnorm=False, use_tanh=False):
        """
        img_shape - the size of the input data. Shape = (..., C, H, W)
        """
        super(DCGAN, self).__init__()
        self.img_shape = img_shape

        self.discriminator = Discriminator(img_shape, bnorm=disc_bnorm, virt_bnorm=vbnorm)
        self.generator = Generator(img_shape, z_size=z_size, trainable_z=trainable_z, bnorm=gen_bnorm, use_tanh=use_tanh)

    def discriminate(self, x):
        return self.discriminator(x)

    def generate(self, n_samples):
        return self.generator(n_samples)

    def clip_ds_ps(self, clip_coef):
        for p in self.discriminator.parameters():
            p.data.clamp_(-clip_coef, clip_coef)

