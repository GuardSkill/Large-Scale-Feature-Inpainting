###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]
        # slide windows size

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialModule(nn.Module):
    """
    Partial Convlution layer (default activation:LeakyReLU)
    Params: multi_channel, return_mask ,others are same as conv2d.
    Input: The feature image and mask map
    Output:Feature and mask in next layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activ='leaky', instance_norm=False, spectral_norm=True, multi_channel=True, return_mask=True):
        super().__init__()
        self.conv = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  padding=padding,
                                  stride=stride, dilation=dilation, bias=bias, multi_channel=multi_channel,
                                  return_mask=return_mask)

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.spectral_norm = True
        if instance_norm:
            self.bn = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class GatedModule(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activ='leaky', instance_norm=False, spectral_norm=True):
        super(GatedModule, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        if spectral_norm:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
            self.mask_conv2d = nn.utils.spectral_norm(self.mask_conv2d)
            self.spectral_norm = True
        if instance_norm:
            self.bn = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.sigmoid(self.mask_conv2d(input))
        if hasattr(self, 'activation'):
            h = self.activation(x) * mask
        else:
            h = x * mask
        if hasattr(self, 'bn'):
            h = self.bn(h)
        return h


class DeConvGatedModule(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
    #              activ='leaky',instance_norm=False,spectral_norm=True):

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, activ='leaky', instance_norm=True, spectral_norm=True):
        super(DeConvGatedModule, self).__init__()
        self.conv2d = GatedModule(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                  activ, instance_norm, spectral_norm)
        self.scale_factor = scale_factor

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, self.scale_factor)
        return self.conv2d(x)


class SEModule(nn.Module):
    """
    Squeeze-Excitation Module
    Params:
        n_features(int):  the number of input channels
        reduction (int, optional): the number of input units/middle units
    Input: The feature from last layer "I"
    Output:Squeeze and Excitation output
    """

    def __init__(self, n_features,spectral_norm=True, reduction=16):
        super(SEModule, self).__init__()
        if n_features %reduction ==3:
            reduction=1
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        if spectral_norm:
            self.linear1=nn.utils.spectral_norm(self.linear1)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        if spectral_norm:
            self.linear2=nn.utils.spectral_norm(self.linear2)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class CFSModule(nn.Module):
    """
    Comprehensive Feature Selection Convolution
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activ='leaky', instance_norm=False, spectral_norm=True):
        super(CFSModule, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.same_conv2d = torch.nn.Conv2d(in_channels, in_channels, kernel_size, 1, int((kernel_size - 1)* dilation/ 2),
                                           dilation, groups,
                                           bias)
        self.sqex = SEModule(out_channels,spectral_norm)
        self.sigmoid = torch.nn.Sigmoid()
        if spectral_norm:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
            self.mask_conv2d = nn.utils.spectral_norm(self.mask_conv2d)
            self.same_conv2d = nn.utils.spectral_norm(self.same_conv2d)
            self.spectral_norm = True
        if instance_norm:
            self.bn = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        same_mask = self.sigmoid(self.same_conv2d(input))
        x = self.conv2d(same_mask * input)
        mask = self.sigmoid(self.mask_conv2d(input))
        if hasattr(self, 'activation'):
            h = self.activation(x) * mask
            h = self.sqex(h)
        else:
            h = x * mask            # no activation and no SE in last layer
        if hasattr(self, 'bn'):
            h = self.bn(h)
        return h


# Residual block using Squeeze and Excitation

class ResCFSBlock(nn.Module):

    def __init__(self, dim, dilation=2):
        super(ResCFSBlock, self).__init__()

        # CFS convolutions
        self.conv1 = CFSModule(in_channels=dim, out_channels=dim, kernel_size=3,
                               padding=int((3 - 1) * dilation / 2),
                               dilation=dilation, bias=False)
        self.conv2 = CFSModule(in_channels=dim, out_channels=dim, kernel_size=3,
                               padding=int((3 - 1) * dilation / 2),
                               dilation=dilation, bias=False)

    def forward(self, x):
        # convolutions
        h = self.conv1(x)
        h = self.conv2(h)

        # add residuals
        h = torch.add(x, h)

        return h


class PartialResnetBlock(nn.Module):
    def __init__(self, dim, dilation=2):
        super(PartialResnetBlock, self).__init__()
        # (in_channels=512 + 256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv_block1 = PartialModule(in_channels=dim, out_channels=dim, kernel_size=3,
                                         padding=int((3 - 1) * dilation / 2),
                                         dilation=dilation, bias=False, spectral_norm=True)
        self.conv_block2 = PartialModule(in_channels=dim, out_channels=dim, kernel_size=3,
                                         padding=int((3 - 1) * dilation / 2),
                                         dilation=dilation, bias=False, spectral_norm=True)

    def forward(self, comp):
        [x, mask] = comp
        h, h_mask = self.conv_block1(x, mask)
        h, h_mask = self.conv_block2(h, h_mask)
        out = torch.mul(x, mask) + h

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return [out, h_mask]
