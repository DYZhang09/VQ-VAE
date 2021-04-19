import torch
import torch.nn as nn

from model.basic import ResBlockStack, ConvTransposed2dStack


##########################################################
# This file defines the decoder used in VQVAE            #
##########################################################

class Decoder(nn.Module):
    """
    VQVAE decoder
    """

    def __init__(self,
                 in_nc,
                 hidden_nc,
                 out_nc):
        super().__init__()
        blocks = [
            ResBlockStack(in_nc,
                          kernel_size=[3, 1],
                          padding=[1, 0],
                          n_res_layer=2,
                          activation_first=True,
                          n_blocks=2),
            ConvTransposed2dStack(in_nc, hidden_nc, out_nc,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  n_layers=2)
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 128, 32, 32
    decoder = Decoder(in_nc=C, hidden_nc=256, out_nc=3)
    test_vec = torch.randn(N, C, H, W)
    test_out = decoder(test_vec)
    print(test_out.shape)