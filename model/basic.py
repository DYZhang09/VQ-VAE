import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################
# This file defines the basic layers used in VQVAE      #
#########################################################

class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_nc):
        super().__init__()
        blocks = [
            nn.ReLU(),
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=1, stride=1, padding=0)
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return input + self.net(input)


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

        self.ver2hor_conv = nn.Conv2d(in_nc * 2, in_nc * 2, 1)

        kernel_sz = (1, kernel_size // 2 + 1)
        pad_sz = (0, kernel_size // 2)
        self.horizontal_conv = nn.Conv2d(in_nc, 2 * in_nc, kernel_sz, 1, pad_sz)

        self.hor_out_conv = nn.Conv2d(in_nc, in_nc, 1)

        self.gate = GatedActivation()

    def get_causal(self):
        self.vertical_conv.weight.data[:, :, -1].zero_()
        self.horizontal_conv.weight.data[:, :, :, -1].zero_()

    def forward(self, x_h, x_v, h):
        if self.make_causal:
            self.get_causal()

        h = self.cond_embed(h)
        x_vert = self.vertical_conv(x_v)
        x_vert = x_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(x_vert + h[:, :, None, None])

        x_hor = self.horizontal_conv(x_h)
        x_hor = x_hor[:, :, :, :x_h.size(-2)]
        x_hor += self.ver2hor_conv(x_vert)

        out_h = self.gate(x_hor + h[:, :, None, None])
        out_h = self.hor_out_conv(out_h)
        if self.residual:
            out_h += x_h
        return out_v, out_h


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 3, 128, 128
