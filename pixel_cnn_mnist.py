import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model.gatedPixelCNN import GatedPixelCNN
from dataIO.utils import *


##########################################################
# This file is used to train the GatedPixelCNN on MNIST  #
# dataset to test the PixelCNN implementation            #
##########################################################

def get_mnist_train_loader(batch=32):
    return data.DataLoader(datasets.MNIST(r'./datasets', transform=transforms.ToTensor(), download=True),
                           batch)


def get_mnist_test_loader(batch=32):
    return data.DataLoader(datasets.MNIST(r'./datasets', transform=transforms.ToTensor(), download=True),
                           batch)


def train(model: nn.Module, criterion, optimizer, device, epochs, dataloader):
    for epoch in range(epochs):
        for x, label in dataloader:
            x = (x[:, 0] * 255).long().to(device)
            label = label.to(device)

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = criterion(logits.view(-1, 256), x.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("|epoch: %d, loss: %.6f" % (epoch, loss.item()))
            path = os.path.join(r'./weights', 'pixelcnn_mnist_epoch_%d.pth' % epoch)
            torch.save(model.state_dict(), path)


def generate(model, device):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().to(device)

    x_tilde = model.generate(label=label, batch_size=100, out_size=(28, 28))
    out = x_tilde.cpu().data.float()
    print(out.max())

    print(out.shape)
    N = out.shape[0]
    for i in range(N):
        out_img = tensor2img(out[i].unsqueeze(0).repeat(3, 1, 1))
        write_image(os.path.join(r'./out',
                                 "pixelcnn_mnist_samples_%d.jpg" % i),
                    out_img)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default='cpu', help='the device')
    args.add_argument("--epochs", type=int, default=0)
    args.add_argument("--batch", type=int, default=64)
    args.add_argument("--weight", type=str, default=None)
    args = args.parse_args()

    train_loader = get_mnist_train_loader(batch=args.batch)

    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = GatedPixelCNN(in_nc=256, hidden_dim=64, device=device)
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train(model, criterion, optimizer, device, args.epochs, train_loader)
    generate(model, device)
