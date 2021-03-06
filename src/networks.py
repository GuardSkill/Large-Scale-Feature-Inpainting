import os

import torch
import torch.nn as nn
from libs.Modules import PartialModule, GatedModule, DeConvGatedModule, PartialResnetBlock, CFSModule, ResCFSBlock
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.001):
        # def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0
        /models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, std=gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


# receptive field fusion network
class RFFNet(BaseNetwork):
    def __init__(self, in_channels=3, number_b=4, init_weights=True):
        self.inplanes = in_channels
        super(RFFNet, self).__init__()

        # stem net

        # get 32 dimention feature
        self.layer1 = self._make_layer(3, 32, stride=1)

        in_channels = [32]
        num_blocks = [number_b]
        out_channels = [32, 64]
        # out_channels = [64, 64]
        self.stage0, pre_stage_channels = self._make_stage(
            num_blocks, in_channels, out_channels, stage_index=0)

        out_channels = [32, 128]
        # out_channels = [64, 128]
        num_blocks = [number_b, number_b]

        self.stage1, pre_stage_channels = self._make_stage(
            num_blocks, pre_stage_channels, out_channels, stage_index=1)

        # out_channels = [32, 256]
        out_channels = [32, 256]
        num_blocks = [number_b, number_b]

        self.stage2, pre_stage_channels = self._make_stage(
            num_blocks, pre_stage_channels, out_channels, stage_index=2)

        # out_channels = [32, 512]
        out_channels = [32, 512]
        num_blocks = [number_b, number_b]

        self.stage3, pre_stage_channels = self._make_stage(
            num_blocks, pre_stage_channels, out_channels, stage_index=3)

        up_factor = 2 << 3

        sub_pixel = []
        temp_c = out_channels[1]
        for _ in range(4):
            sub_pixel.append(
                nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(
                            temp_c,
                            int(temp_c / 2 * 2 ** 2),  # int(temp_c/2*2**2) (channel_num/2) * factor**2
                            3, 1, 1, bias=False
                        ), True),
                    # nn.LeakyReLU(0.2, inplace=False),
                    # nn.Tanh(),
                    nn.PixelShuffle(2),
                    nn.Tanh()
                )
            )
            temp_c = temp_c // 2

        self.up_layer = nn.Sequential(*sub_pixel)

        self.final_layer = nn.Sequential(spectral_norm(nn.Conv2d(
            in_channels=out_channels[0] * 2,
            out_channels=out_channels[0],
            kernel_size=7,
            stride=1,
            padding=3
        ), True),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            spectral_norm(nn.Conv2d(
                in_channels=out_channels[0],
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1), True))

        if init_weights:
            self.init_weights()

    def _make_layer(self, in_channel, out_channel, stride=1):
        downsample = None
        if stride != 1 or in_channel != out_channel:
            downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                                   padding=0, bias=False)
        layer = BottleneckBlock(in_channel, out_channel, stride=1, dilation=1, use_spectral_norm=True,
                                downsample=downsample)
        return layer

    def _make_stage(self, num_blocks, in_channels, out_channel, multi_scale_output=True, stage_index=0):
        num_blocks = num_blocks
        fuse_method = 'Concatenate'
        stage = TwoBranchModule(
            num_blocks,
            in_channels,
            out_channel,
            fuse_method,
            True,
            stage_index
        )

        out_channels = stage.get_out_channels()

        return stage, out_channels

    def forward(self, x):
        x = self.layer1(x)
        x_list = self.stage0([x])
        x_list = self.stage1(x_list)
        x_list = self.stage2(x_list)
        x_list = self.stage3(x_list)
        x_list[1] = self.up_layer(x_list[1])
        x = torch.cat(x_list, 1)
        x = self.final_layer(x)
        x = (torch.tanh(x) + 1) / 2
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# Basic Block of residual network
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                    padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            nn.ZeroPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                    padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # out = nn.LeakyReLU(0.2, inplace=False)(out)
        out = nn.Tanh()(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, use_spectral_norm=False, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(dilation),
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2, kernel_size=3, stride=stride,
                          padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            nn.ZeroPad2d(1),
            spectral_norm(
                nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=stride,
                          padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv_block(x) + residual
        # out = nn.LeakyReLU(0.2, inplace=False)(out)
        out = nn.Tanh()(out)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class TwoBranchModule(nn.Module):
    def __init__(self, num_blocks, in_channels,
                 out_channels, fuse_method, multi_scale_output=True, stage_index=0):
        super(TwoBranchModule, self).__init__()
        self.stage = stage_index
        if self.stage > 0:
            self.num_branches = 2
        else:
            self.num_branches = 1
        # check num_blocks num_channels num_inchannels if have 2 for 2 branches
        self._check_branches(self.num_branches, num_blocks, in_channels, out_channels)
        self.num_inchannels = in_channels
        self.out_channels = out_channels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_blocks)
        self.fuse_layers = self._make_fuse_layers(out_channels)
        self.fused_branches = self._make_dimention_reduce(out_channels)

    def _make_one_branch(self, branch_index, num_blocks,
                         dilation):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(
                ResnetBlock(
                    self.num_inchannels[branch_index], dilation=dilation, use_spectral_norm=True
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_blocks):
        branches = []
        # main branch
        branches.append(
            self._make_one_branch(0, num_blocks, 1)
        )
        # vice branch :dilation ==2
        if self.num_branches == 2:
            branches.append(
                self._make_one_branch(1, num_blocks, 2)
            )

        return nn.ModuleList(branches)

    def _make_d_reduce(self, branch_index, out_channels,
                       dilation=1):
        layers = []
        # stage >0, need concatenate
        if self.stage > 0:
            if branch_index == 0:
                in_c = self.num_inchannels[0] + self.out_channels[0]
            else:
                in_c = self.num_inchannels[0] * (1 << (self.stage + 1)) + self.num_inchannels[1] * 2
        else:
            if branch_index == 0:
                in_c = self.num_inchannels[0]
            else:
                in_c = self.num_inchannels[0] * (self.stage + 1) + self.num_inchannels[0]
        for i in range(1):
            layers.append(
                nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(in_channels=in_c, out_channels=out_channels[branch_index],
                                  kernel_size=3, stride=1,
                                  padding=1, dilation=dilation, bias=not True), True),
                    # nn.LeakyReLU(0.2, inplace=False)
                    nn.Tanh()
                ))

        return nn.Sequential(*layers)

    def _make_dimention_reduce(self, out_channels):
        branches = []
        # main branch
        branches.append(
            self._make_d_reduce(0, out_channels)
        )
        # vice branch :dilation ==2
        branches.append(
            self._make_d_reduce(1, out_channels)
        )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, out_channels):
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers = []
        for i in range(2 if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                #  smaller scale to  biger scale
                if j > i:
                    sub_pixel = []
                    temp_c = num_inchannels[j]
                    for _ in range(self.stage):
                        sub_pixel.append(
                            nn.Sequential(
                                spectral_norm(
                                    nn.Conv2d(
                                        temp_c,
                                        temp_c * 2,
                                        3, 1, 1, bias=False
                                    ), True),
                                # nn.LeakyReLU(0.2, inplace=False),
                                # nn.Tanh(),
                                nn.PixelShuffle(2),
                                nn.Tanh()
                            )
                        )
                        temp_c = temp_c // 2
                    fuse_layer.append(nn.Sequential(*sub_pixel))

                # main branch
                elif j == i == 0:
                    fuse_layer.append(None)
                #  bigger scale to smaller scale
                elif j < i:
                    # sequential strided conv
                    temp_c = num_inchannels[j]
                    conv3x3s = []

                    # get the stride between the two stream
                    for s in range(self.stage + 1):
                        if s == self.stage:  # last conv layer
                            conv3x3s.append(
                                nn.Sequential(
                                    spectral_norm(nn.Conv2d(
                                        temp_c,
                                        # out_channels[i],
                                        temp_c * 2,
                                        3, 2, 1, bias=False
                                    ), True),
                                    # nn.LeakyReLU(0.2, inplace=False)
                                    nn.Tanh()
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    spectral_norm(nn.Conv2d(
                                        temp_c,
                                        temp_c * 2,
                                        3, 2, 1, bias=False
                                    ), True),
                                    # nn.LeakyReLU(0.2, inplace=False)
                                    nn.Tanh()
                                )
                            )
                            temp_c = temp_c * 2
                    fuse_layer.append(nn.Sequential(*conv3x3s))

                # vice branch j for vice branch i
                else:
                    fuse_layer.append(
                        nn.Sequential(
                            spectral_norm(nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[j] * 2,
                                3, 2, 1, bias=False
                            ), True),
                            # nn.LeakyReLU(0.2, inplace=False),
                            nn.Tanh()
                        ))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        # interactive module
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            # y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y1 = self.fuse_layers[i][j](x[j])
                y = torch.cat((y, y1), 1)
            x_fuse.append(y)

        for i in range(len(x_fuse)):
            x_fuse[i] = self.fused_branches[i](x_fuse[i])

        return x_fuse

    def _check_branches(self, num_branches, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)


class UnetGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(UnetGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


# Unet network for similar parameter number
class UnetGeneratorSame(BaseNetwork):
    def __init__(self, residual_blocks=4, init_weights=True):
        super(UnetGeneratorSame, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(512, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


