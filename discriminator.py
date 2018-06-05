import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from vbnorm import VirtBatchNorm2d

class Discriminator(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, img_shape, bnorm=False, vbnorm=False):
        """
        img_shape - the size of the input data. Shape = (..., C, H, W)
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.bnorm = bnorm
        self.vbnorm = vbnorm

        # Discriminator
        self.convs = nn.ModuleList([])
        shape = [*self.img_shape[-3:]]

        ksize=3; padding=1; stride=1; out_depth = 32
        self.convs.append(self.conv_block(img_shape[-3], out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth=out_depth
        out_depth = 64
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth = out_depth
        out_depth = 128
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth = out_depth
        out_depth = 256
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth = out_depth
        out_depth = 512
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm, vbnorm=vbnorm))

        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)
        self.features = nn.Sequential(*self.convs)
        self.feat_shape = shape
        self.flat_size = int(np.prod(shape))
        print("Flat size:", self.flat_size)

        # Classifier
        block = []
        block.append(nn.Linear(self.flat_size, 1))
        self.classifier = nn.Sequential(*block)

    def get_new_shape(self, old_shape, depth, ksize=3, stride=1, padding=1):
        new_shape = [depth]
        for i in range(len(old_shape[1:])):
            new_shape.append((old_shape[i+1] - ksize + 2*padding)//stride + 1)
        return new_shape

    def discriminate(self, x):
        fx = self.features(x)
        fx = fx.contiguous().view(len(fx), -1)
        return self.classifier(fx)

    def forward(self, x):
        fx = self.features(x)
        fx = fx.contiguous().view(len(fx), -1)
        return self.classifier(fx)

    def clip_ds_ps(self, clip_coef):
        for p in self.parameters():
            p.data.clamp_(-clip_coef, clip_coef)

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
