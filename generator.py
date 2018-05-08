import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class Generator(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, img_shape, flat_size, feat_shape, z_size=256, trainable_z=False, bnorm=False):
        """
        img_shape - the size of the input data. Shape = (..., C, H, W)
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.flat_size = flat_size
        self.feat_shape = feat_shape
        self.z_size = z_size
        self.trainable_z = trainable_z
        self.bnorm = bnorm

        # Trainable Mean and STD
        self.mu = nn.Parameter(torch.zeros(1,z_size))
        self.std = nn.Parameter(torch.ones(1,z_size))
        self.z_stats = nn.ParameterList([self.mu, self.std])

        # Fully Connected Portion
        block = [nn.Linear(z_size, self.flat_size),nn.ReLU()]
        if bnorm:
            block.append(nn.BatchNorm1d(self.flat_size))
        self.gen_flat = nn.Sequential(*block)

        # Convolution Transpose Portion
        self.deconvs = nn.ModuleList([])

        ksize=5; padding=0; stride=1; in_depth = 128
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=4; padding=0; stride=2; in_depth = out_depth
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 32
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 32
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 16
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=3; padding=0; stride=1; in_depth = out_depth
        out_depth = 16
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, activation=None, bnorm=False))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 8
        self.deconvs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, activation=None, bnorm=False))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = img_shape[-3]
        self.deconvs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, activation=None, bnorm=False))

        self.generator = nn.Sequential(*self.deconvs)

    def generate(self, n_samples):
        zs = Variable(self.cuda_if(torch.randn(n_samples, self.z_size)))
        if self.trainable_z:
            zs = self.std * zs + self.mu
        samples = self.gen_flat(zs)
        samples = samples.view(-1,*self.feat_shape)
        return self.generator(samples)

    def forward(self, n_samples):
        zs = Variable(self.cuda_if(torch.randn(n_samples, self.z_size)))
        if self.trainable_z:
            zs = self.std * zs + self.mu
        samples = self.gen_flat(zs)
        samples = samples.view(-1,*self.feat_shape) 
        return self.generator(samples)

    def deconv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='relu',bnorm=False):
        block = []
        block.append(nn.ConvTranspose2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        if bnorm:
            block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)
        
    def conv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='relu',bnorm=False):
        block = []
        block.append(nn.Conv2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        if bnorm:
            block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)