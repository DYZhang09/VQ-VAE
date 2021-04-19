import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################
# This file defines the vector quantize(VQ) layer used  #
# in VQVAE                                              #
#########################################################


class Quantize(nn.Module):
    """
    Vector Quantize(VQ) layer
    """

    def __init__(self, embed_size, embed_dim):
        super().__init__()
        self.embed_size = embed_size
        self.embed_dim = embed_dim

        self.register_buffer("embed", torch.randn(embed_size, embed_dim))

    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed)

    def forward(self, input: torch.Tensor):
        assert input.shape[1] == self.embed_dim, "input dim doesn't match the dim of codebook!\n"
        N, C, H, W = input.shape
        flatten = input.permute([0, 2, 3, 1]).contiguous().view(-1, self.embed_dim).type(torch.float)
        dist = torch.functional.cdist(flatten, self.embed)
        _, embed_idx = dist.min(dim=1)
        embed_idx = embed_idx.view(N, H, W)
        quantize = self.embed_code(embed_idx).permute([0, 3, 1, 2])

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, embed_idx, diff


# unit test
if __name__ == '__main__':
    N, C, H, W = 100, 128, 32, 32
    test_vec = torch.randn(N, C, H, W)
    quantize = Quantize(embed_size=512, embed_dim=C)
    test_out, idx, diff = quantize(test_vec)
    print(test_out.shape, idx.shape, diff.shape)