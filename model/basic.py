import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################
# This file defines the basic layers used in VQVAE      #
#########################################################


class Conv2dStack(nn.Module):
    """
    stacked conv2d block
    """

    def __init__(self,
                 in_nc: int,
                 hidden_nc: int,
                 out_nc: int,
                 kernel_size=4,
                 stride=1,
                 padding=0,
                 n_layers: int = 2,
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * n_layers
        if not isinstance(stride, list):
            stride = [stride] * n_layers
        if not isinstance(padding, list):
            padding = [padding] * n_layers

        blocks = []
        for i in range(n_layers):
            in_channel = in_nc if i == 0 else hidden_nc
            out_channel = hidden_nc if i != n_layers - 1 else out_nc
            blocks.append(nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=kernel_size[i],
                                    stride=stride[i],
                                    padding=padding[i]))
            if i != n_layers - 1:
                blocks.append(nn.BatchNorm2d(out_channel))
            blocks.append(activation)
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


class ConvTransposed2dStack(nn.Module):
    """
    stacked transposed conv2d block
    """

    def __init__(self,
                 in_nc,
                 hidden_nc,
                 out_nc,
                 kernel_size=4,
                 stride=1,
                 padding=0,
                 n_layers=1,
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * n_layers
        if not isinstance(stride, list):
            stride = [stride] * n_layers
        if not isinstance(padding, list):
            padding = [padding] * n_layers

        blocks = []
        for i in range(n_layers):
            in_channel = in_nc if i == 0 else hidden_nc
            out_channel = hidden_nc if i != n_layers - 1 else out_nc
            blocks.append(nn.ConvTranspose2d(in_channel, out_channel,
                                             kernel_size=kernel_size[i],
                                             stride=stride[i],
                                             padding=padding[i]))
            blocks.append(activation)
            if i != n_layers - 1:
                blocks.append(nn.BatchNorm2d(out_channel))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self,
                 in_nc: int,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 n_layers: int = 1,
                 activation: nn.Module = nn.ReLU(),
                 activation_first: bool = False):
        super().__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * n_layers
        if not isinstance(stride, list):
            stride = [stride] * n_layers
        if not isinstance(padding, list):
            padding = [padding] * n_layers

        blocks = []
        for i in range(n_layers):
            if activation_first:
                blocks.append(activation)
                blocks.append(nn.Conv2d(in_nc, in_nc,
                                        kernel_size=kernel_size[i],
                                        stride=stride[i],
                                        padding=padding[i]))
            else:
                blocks.append(nn.Conv2d(in_nc, in_nc,
                                        kernel_size=kernel_size[i],
                                        stride=stride[i],
                                        padding=padding[i]))
                blocks.append(activation)
            if i != n_layers - 1:
                blocks.append(nn.BatchNorm2d(in_nc))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return input + self.net(input)


class ResBlockStack(nn.Module):
    """
    stacked residual blocks
    """

    def __init__(self,
                 in_nc: int,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 n_res_layer=1,
                 activation: nn.Module = nn.ReLU(),
                 activation_first: bool = False,
                 n_blocks: int = 1):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            blocks.append(ResBlock(in_nc,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   n_layers=n_res_layer,
                                   activation=activation,
                                   activation_first=activation_first))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


class GatedActivation(nn.Module):
    """
    gated activation layer
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        x, y = input.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedPixelCNNConv2d(nn.Module):
    """
    a single layer in the Gated PixelCNN architecture
    """

    def __init__(self, in_nc, kernel_size, residual=True, num_class=10, make_causal=False):
        super().__init__()
        assert kernel_size % 2 == 1, "the [kernel_size] must be odd"

        self.residual = residual
        self.make_causal = make_causal

        self.cond_embed = nn.Embedding(num_class, in_nc * 2)

        kernel_sz = (kernel_size // 2 + 1, kernel_size)
        pad_sz = (kernel_size // 2, kernel_size // 2)
        self.vertical_conv = nn.Conv2d(in_nc, in_nc * 2, kernel_sz, 1, pad_sz)

        self.ver2hor_conv = nn.Conv2d(in_nc * 2, in_nc*2, 1)

        kernel_sz = (1, kernel_size // 2 + 1)
        pad_sz = (0, kernel_size // 2)
        self.horizontal_conv = nn.Conv2d(in_nc, 2 * in_nc, kernel_sz, 1, pad_sz)

        self.hor_out_conv = nn.Conv2d(in_nc, in_nc, 1)

        self.gate = GatedActivation()

        self.x_h = None
        self.x_v = None
        self.h = None

    def set_input(self, x_h, x_v, h):
        self.x_h = x_h
        self.x_v = x_v
        self.h = h

    def get_causal(self):
        self.vertical_conv.weight.data[:, :, -1].zero_()
        self.horizontal_conv.weight.data[:, :, :, -1].zero_()

    def forward(self):
        if self.make_causal:
            self.get_causal()

        h = self.cond_embed(self.h)
        x_vert = self.vertical_conv(self.x_v)
        x_vert = x_vert[:, :, :self.x_v.size(-1), :]
        out_v = self.gate(x_vert + h[:, :, None, None])

        x_hor = self.horizontal_conv(self.x_h)
        x_hor = x_hor[:, :, :, :self.x_h.size(-2)]
        x_hor += self.ver2hor_conv(x_vert)

        out_h = self.gate(x_hor + h[:, :, None, None])
        out_h = self.hor_out_conv(out_h)
        if self.residual:
            out_h += self.x_h
        return out_v, out_h


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 3, 128, 128
    resblock = ResBlock(in_nc=3, kernel_size=[3, 1], stride=1, padding=[1, 0], n_layers=2, activation_first=True)
    test_vec = torch.randn(N, C, H, W)
    test_out = resblock(test_vec)
    print(test_out)

    resstack = ResBlockStack(in_nc=C, kernel_size=[3, 1], stride=1, padding=[1, 0], n_res_layer=2,
                             activation_first=True,
                             n_blocks=2)
    test_out = resstack(test_vec)
    print(test_out.shape)

    convstack = Conv2dStack(in_nc=C, hidden_nc=256, out_nc=1,
                            kernel_size=4, stride=2, padding=1, n_layers=2)
    test_out = convstack(test_vec)
    print(test_out.shape)

    transstack = ConvTransposed2dStack(in_nc=1, hidden_nc=256, out_nc=C,
                                       kernel_size=4, stride=2, padding=1,
                                       n_layers=2)
    test_out = transstack(test_out)
    print(test_out.shape)
