import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


################################################################
# This file defines the util classes for getting data from     #
# datasets                                                     #
################################################################


class CIFAR10(object):
    """
    CIFAR-10 dataset
    """

    def __init__(self, root_path, batch_size, shuffle=False, transform=None, download=True, num_workers=1):
        if transform is None:
            transform = [transforms.ToTensor()]
            transform = transforms.Compose(transform)
        dataset_train = datasets.CIFAR10(root_path, train=True, transform=transform, download=download)
        dataset_test = datasets.CIFAR10(root_path, train=False, transform=transform, download=download)
        self.train_loader = data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)
        self.test_loader = data.DataLoader(dataset_test,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)

    def get_loader(self):
        return self.train_loader, self.test_loader


class CIFAR100(object):
    """
    CIFAR-100 dataset
    """

    def __init__(self, root_path, batch_size, shuffle=False, transform=None, download=True, num_workers=1):
        if transform is None:
            transform = [transforms.ToTensor()]
            transform = transforms.Compose(transform)
        dataset_train = datasets.CIFAR100(root_path, train=True, transform=transform, download=download)
        dataset_test = datasets.CIFAR100(root_path, train=False, transform=transform, download=download)
        self.train_loader = data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)
        self.test_loader = data.DataLoader(dataset_test,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)

    def get_loader(self):
        return self.train_loader, self.test_loader


# unit test
if __name__ == '__main__':
    cifar10 = CIFAR10(r'../dataset', batch_size=64, shuffle=True, download=True, num_workers=2)
    train_loader, test_loader = cifar10.get_loader()
    for i, data in enumerate(test_loader):
        print(data[0].shape, data[1].shape)
