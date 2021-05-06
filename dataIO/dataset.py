import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

from dataIO.utils import read_image
from dataIO.dataspliter import DataSpliter


################################################################
# This file defines the util classes for getting data from     #
# datasets                                                     #
################################################################

class CustomDataset(data.Dataset):
    """
    custom dataset
    """

    def __init__(self,
                 paths,
                 transform=None):
        super(CustomDataset, self).__init__()
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = self.paths[item]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, img_path)
