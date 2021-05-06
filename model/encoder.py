import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic import ResBlock


############################################################
# This file defines the encoder used in VQVAE              #
############################################################


class Encoder(nn.Module):
    """
    VQVAE encoder
    """

    def __init__(self,
                 in_nc,
                 out_nc):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels=in_nc,
                      out_channels=out_nc,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_nc,
                      out_channels=out_nc,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            ResBlock(out_nc),
            ResBlock(out_nc)
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 3, 32, 32
    test_vec = torch.randn(N, C, H, W)
    encoder = Encoder(in_nc=3, out_nc=128)
    test_out = encoder(test_vec)
    print(test_out.shape)
