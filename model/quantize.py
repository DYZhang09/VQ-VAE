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
        self.embed = nn.Embedding(embed_size, embed_dim)
        self.embed.weight.data.uniform_(-1.0 / embed_size, 1.0 / embed_size)

    def embed_code(self, embed_idx):
        return self.embed(embed_idx)

    def forward(self, input: torch.Tensor):
        assert input.shape[1] == self.embed_dim, "input dim doesn't match the dim of codebook!\n"
        N, C, H, W = input.shape

        # input = torch.randn(1, 128, 64, 64)
        flatten = input.permute([0, 2, 3, 1]).contiguous().view(-1, self.embed_dim).type(torch.float)
        dist = torch.cdist(flatten, self.embed.weight.data)
        _, embed_idx = dist.min(dim=1)
        # print(embed_idx)
        embed_idx = embed_idx.view(N, H, W)
        quantize = self.embed_code(embed_idx)
        quantize = quantize.permute([0, 3, 1, 2])
        # print(quantize)

        embed_loss = F.mse_loss(input.detach(), quantize)
        commit_loss = F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        return quantize, embed_idx, embed_loss, commit_loss


# unit test
if __name__ == '__main__':
    a = torch.randn(4, 8)
    b = torch.randn(10, 8)
    print(b)
    dist = torch.cdist(a, b)
    print(dist)
    m, idx = dist.min(1)
    print(m, idx, F.embedding(idx, b))
    print(b[idx])
