import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic import ResBlockStack, Conv2dStack, ConvTransposed2dStack


############################################################
# This file defines the encoder used in VQVAE              #
############################################################


class Encoder(nn.Module):
    """
    VQVAE encoder
    """

    def __init__(self,
                 in_nc,
                 hidden_nc,
                 out_nc):
        super().__init__()
        blocks = [
            Conv2dStack(in_nc, hidden_nc, hidden_nc,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        n_layers=2,
                        activation=nn.ReLU()),
            ResBlockStack(hidden_nc,
                          kernel_size=[3, 1],
                          stride=1,
                          padding=[1, 0],
                          n_res_layer=2,
                          n_blocks=2),
            nn.Conv2d(hidden_nc, out_nc, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 3, 128, 128
    test_vec = torch.randn(N, C, H, W)
    encoder = Encoder(in_nc=3, hidden_nc=256, out_nc=128)
    test_out = encoder(test_vec)
    print(test_out.shape)
