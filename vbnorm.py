import torch
import torch.nn as nn

class VirtBatchNorm1d(nn.Module):
    def __init__(self, input_size, eps=1e-7):
        super(VirtBatchNorm1d, self).__init__()
        """
        input_size - a python sequence of the size of the data that will be used
                     shape = (...,C)
        """
        if type(input_size) == type(int()):
            input_size = [input_size]
        self.input_size = input_size
        self.scalers = nn.Parameter(torch.ones(1,input_size[-1]))
        self.shifters = nn.Parameter(torch.zeros(1,input_size[-1]))
        self.eps = 1e-7

    def forward(self, x): 
        """
        x - torch FloatTensor Variable in which the first half of the samples
            should be the virtual batch and the latter half should be the real
            batch of data
            shape = (2*BatchSize, C)
        """

        virtual_batch = x[:len(x)//2]
        means = virtual_batch.mean(0)
        means_sq = virtual_batch.pow(2).mean(0)
        batch_stds = means_sq - means.pow(2) 

        x = (x + means) / (batch_stds.sqrt() + self.eps)
        x = x*self.scalers + self.shifters
        return x

class VirtBatchNorm2d(nn.Module):
    def __init__(self, input_size, eps=1e-7):
        super(VirtBatchNorm2d, self).__init__()
        """
        input_size - integer denoting the number of channels of the data that will be used
        """
        self.input_size = input_size
        self.scalers = nn.Parameter(torch.ones(1,input_size))
        self.shifters = nn.Parameter(torch.zeros(1,input_size))
        self.eps = 1e-7

    def forward(self, x): 
        """
        x - torch FloatTensor Variable in which the first half of the samples
            should be the virtual batch and the latter half should be the real
            batch of data
            shape = (2*BatchSize, C, H, W)
        """

        virtual_batch = x[:len(x)//2]
        means = virtual_batch.mean(0)
        means_sq = virtual_batch.pow(2).mean(0)
        batch_stds = means_sq - means.pow(2) 

        x = (x + means) / (batch_stds.sqrt() + self.eps)
        x = x.permute(0,2,3,1)*self.scalers + self.shifters
        return x.permute(0,3,1,2)
