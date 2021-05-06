import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quantize import Quantize
from model.encoder import Encoder
from model.decoder import Decoder


#########################################################
# This file defines the architecture of VQVAE           #
#########################################################


class VQVAE(nn.Module):
    """
    the VQVAE
    """

    def __init__(self,
                 in_nc=3,
                 embed_size=512,
                 embed_dim=128,
                 commit_loss_weight=1):
        super().__init__()
        self.encoder = Encoder(in_nc=in_nc, out_nc=embed_dim)
        self.quantize = Quantize(embed_size=embed_size, embed_dim=embed_dim)
        self.decoder = Decoder(in_nc=embed_dim, out_nc=in_nc)
        self.commit_loss_weight = commit_loss_weight
        self.z_q = None
        self.z_e = None
        self.q = None
        self.embed_loss = None
        self.commit_loss = None
        self.out = None
        self.real_data = None

    def forward(self, real_data):
        self.real_data = real_data
        self.z_e = self.encoder(real_data)
        self.z_q, self.q, self.embed_loss, self.commit_loss = self.quantize(self.z_e)
        self.embed_loss = self.embed_loss.unsqueeze(0)
        self.out = self.decoder(self.z_q)
        return self.out

    def calc_loss(self):
        recon_loss = F.mse_loss(self.out, self.real_data)
        return recon_loss + self.embed_loss + self.commit_loss_weight * self.commit_loss

    def evaluate(self, data):
        with torch.no_grad():
            self.forward(data)
            return self.calc_loss()

    def test(self, data):
        with torch.no_grad():
            return self.forward(data)


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 3, 128, 128
    test_vec = torch.randn(N, C, H, W)
    vqvae = VQVAE()
    out = vqvae(test_vec)
    print(out.shape)
    loss = vqvae.calc_loss()
    print(loss)
    loss.backward()

