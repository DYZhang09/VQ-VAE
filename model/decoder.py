import torch
import torch.nn as nn

from model.basic import ResBlock


##########################################################
# This file defines the decoder used in VQVAE            #
##########################################################

class Decoder(nn.Module):
    """
    VQVAE decoder
    """

    def __init__(self,
                 in_nc,
                 out_nc):
        super().__init__()
        blocks = [
            ResBlock(in_nc),
            ResBlock(in_nc),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_nc,
                               out_channels=in_nc,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_nc,
                               out_channels=out_nc,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 128, 8, 8
    decoder = Decoder(in_nc=C, out_nc=3)
    test_vec = torch.randn(N, C, H, W)
    test_out = decoder(test_vec)
    print(test_out.shape)
