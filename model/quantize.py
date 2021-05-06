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
        self.embed.weight.data.uniform_(-1.0, 1.0)

    def embed_code(self, embed_idx):
        return self.embed(embed_idx)

    def forward(self, input: torch.Tensor):
        assert input.shape[1] == self.embed_dim, "input dim doesn't match the dim of codebook!\n"
        N, C, H, W = input.shape

        # print(input)
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
    codebook = nn.Embedding(num_embeddings=3, embedding_dim=2)
    codebook.weight.data.uniform_(0, 10)
    test_vec = torch.randint(high=10, size=(1, 2)).type(torch.float)
    print(codebook.weight.data)
    print(test_vec)
    dist = torch.cdist(test_vec, codebook.weight.data)
    print(dist)
    _, idx = dist.min(dim=1)
    print(idx)
    print(codebook(idx))

