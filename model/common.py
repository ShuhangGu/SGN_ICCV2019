import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class ImageDownsample(nn.Conv2d):
    def __init__(self, n_colors, scale):
        super(ImageDownsample, self).__init__(n_colors, n_colors, kernel_size=2*scale, bias = False, padding=scale//2,stride=scale)
        kernel_size=2*scale
        self.weight.data = torch.zeros(n_colors,n_colors,kernel_size,kernel_size)
        for i in range(n_colors):
            self.weight.data[i,i,:,:] = torch.ones(1,1,kernel_size,kernel_size)/(kernel_size*kernel_size)
        self.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False





class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SimpleUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):

        m = []
        m.append(conv(n_feat, scale*scale*3, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleUpsampler, self).__init__(*m)

class SimpleGrayUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):

        m = []
        m.append(conv(n_feat, scale*scale, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleGrayUpsampler, self).__init__(*m)

def DownSamplingShuffle(x, scale=4):

    [N,C,W,H] = x.shape
    if(scale==4):
        x1 = x[:,:,0:W:4,0:H:4]
        x2 = x[:,:,0:W:4,1:H:4]
        x3 = x[:,:,0:W:4,2:H:4]
        x4 = x[:,:,0:W:4,3:H:4]
        x5 = x[:,:,1:W:4,0:H:4]
        x6 = x[:,:,1:W:4,1:H:4]
        x7 = x[:,:,1:W:4,2:H:4]
        x8 = x[:,:,1:W:4,3:H:4]
        x9 = x[:,:,2:W:4,0:H:4]
        x10 = x[:,:,2:W:4,1:H:4]
        x11 = x[:,:,2:W:4,2:H:4]
        x12 = x[:,:,2:W:4,3:H:4]
        x13 = x[:,:,3:W:4,0:H:4]
        x14 = x[:,:,3:W:4,1:H:4]
        x15 = x[:,:,3:W:4,2:H:4]
        x16 = x[:,:,3:W:4,3:H:4]
        return torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16),1)
    else:
        x1 = x[:,:,0:W:2,0:H:2]
        x2 = x[:,:,0:W:2,1:H:2]
        x3 = x[:,:,1:W:2,0:H:2]
        x4 = x[:,:,1:W:2,1:H:2]

        return torch.cat((x1,x2,x3,x4),1)


