import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic import GatedPixelCNNConv2d


############################################################
# This file defines the Gated PixelCNN which is used to    #
# VQVAE training(stage 2)                                  #
############################################################


class GatedPixelCNN(nn.Module):
    """
    gated PixelCNN
    """

    def __init__(self,
                 in_nc,
                 hidden_dim,
                 num_gated_block=15,
                 num_class=10):
        super().__init__()
        self.embed_dim = hidden_dim

        self.embedding = nn.Embedding(in_nc, hidden_dim)

        self.layer = nn.ModuleList()
        for i in range(num_gated_block):
            kernel_size = 7 if i == 0 else 3
            self.layer.append(GatedPixelCNNConv2d(hidden_dim, kernel_size, i != 0, num_class, i == 0))

        self.out_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, in_nc, 1)
        )

    def forward(self, input, label):
        shape = input.size() + (-1,)
        x = self.embedding(input.view(-1)).view(shape)
        print(x.shape)
        x = x.permute([0, 3, 1, 2])

        x_v, x_h = (x, x)
        for layer in self.layer:
            layer.set_input(x_v, x_h, label)
            x_v, x_h = layer()

        return self.out_layer(x_h)

    def generate(self, batch_size=64, out_size=(32, 32)):
        x = torch.zeros((batch_size, *out_size)).type(torch.int64)

        for i in range(out_size[0]):
            for j in range(out_size[1]):
                logits = self.forward(x, torch.zeros(x.shape[0], dtype=torch.int64))
                prob = F.softmax(logits[:, :, i, j], dim=-1)
                x[:, :, i, j] = prob.multinomial(1).squeeze().data
        return x


# unit test
if __name__ == '__main__':
   N, C, H, W = 2, 3, 32, 32
   test_vec = torch.zeros(N, H, W).type(torch.int64)
   pixelcnn = GatedPixelCNN(in_nc=3, hidden_dim=32)
   out = pixelcnn(test_vec, torch.zeros(N, dtype=torch.int64))
   print(out.shape)