import torch.nn as nn
from discriminator import Discriminator
from generator import Generator

class GAN(nn.Module):

    def __init__(self, img_shape, z_size=256, trainable_z=True, disc_bnorm=False, gen_bnorm=False):
        """
        img_shape - the size of the input data. Shape = (..., C, H, W)
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.z_size = z_size

        self.discriminator = Discriminator(img_shape, bnorm=disc_bnorm)
        self.flat_size = self.discriminator.flat_size
        self.feat_shape = self.discriminator.feat_shape
        self.generator = Generator(img_shape,self.flat_size, self.feat_shape, z_size=z_size, trainable_z=trainable_z, bnorm=gen_bnorm)

    def discriminate(self, x):
        return self.discriminator(x)

    def generate(self, n_samples):
        return self.generator(n_samples)

    def clip_ds_ps(self, clip_coef):
        for p in self.discriminator.parameters():
            p.data.clamp_(-clip_coef, clip_coef)

