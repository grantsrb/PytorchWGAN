import torch
from torch.autograd import Variable
import torch.nn as nn
from vbnorm import VirtBatchNorm2d
import numpy as np

class Generator(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, img_shape, z_size=100, trainable_z=False, bnorm=False, vbnorm=False, use_tanh=False):
        """
        img_shape - the size of the input data. Shape = (..., C, H, W)
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.z_size = z_size
        self.trainable_z = trainable_z
        self.bnorm = bnorm
        self.vbnorm = vbnorm

        # Trainable Mean and STD
        self.mu = nn.Parameter(torch.zeros(1,z_size))
        self.std = nn.Parameter(torch.ones(1,z_size))
        self.z_stats = nn.ParameterList([self.mu, self.std])

        # Convolution Transpose Portion
        self.deconvs = nn.ModuleList([])

        ksize=4; padding=0; stride=2; in_depth = z_size
        out_depth = 512
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=4; padding=1; stride=2; in_depth = out_depth
        out_depth = 256
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=4; padding=1; stride=2; in_depth = out_depth
        out_depth = 128
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=4; padding=1; stride=2; in_depth = out_depth
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=3; padding=1; stride=1; in_depth = out_depth
        out_depth = self.img_shape[-3]
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=3; padding=1; stride=1; in_depth = out_depth
        out_depth = self.img_shape[-3]
        self.deconvs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        ksize=3; padding=1; stride=1; in_depth = out_depth
        out_depth = self.img_shape[-3]
        if use_tanh:
            self.deconvs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, activation="tanh", bnorm=False, vbnorm=False))
        else:
            self.deconvs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, activation=None, bnorm=False, vbnorm=False))

        self.generator = nn.Sequential(*self.deconvs)

    def generate(self, n_samples, virtual_zs=None):
        zs = Variable(self.get_zs(n_samples))
        if virtual_zs is not None:
            zs = torch.cat([Variable(virtual_zs), zs], dim=0)
        if self.trainable_z:
            zs = self.std * zs + self.mu
        samples = zs.view(-1,self.z_size, 1, 1)
        return self.generator(samples)

    def get_zs(self, n_samples):
        return self.cuda_if(torch.randn(n_samples, self.z_size))

    def forward(self, n_samples, virtual_zs=None):
        return self.generate(n_samples, virtual_zs)

    def deconv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='relu',bnorm=False, vbnorm=False):
        block = []
        block.append(nn.ConvTranspose2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        if bnorm or vbnorm:
            if vbnorm:
                block.append(VirtBatchNorm2d(out_depth))
            else:
                block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)
        
    def conv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='leaky',bnorm=False, vbnorm=False):
        block = []
        block.append(nn.Conv2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        elif activation.lower() == 'leaky':
            block.append(nn.LeakyReLU(negative_slope=0.05))
        if bnorm or vbnorm:
            if vbnorm:
                block.append(VirtBatchNorm2d(out_depth))
            else:
                block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)
