import torch
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms

from dataIO.utils import *
from dataIO.dataset import *


############################################################
# This file defines the dataloader used to load dataset    #
############################################################


class CustomDataloader(object):
    """
    custom dataloader
    """

    def __init__(self,
                 dataset,
                 batch_size=2,
                 shuffle=False,
                 num_workers=1):
        assert isinstance(dataset, data.Dataset), "[dataset] is not a instance of torch.utils.data.Dataset"
        self.dataset = dataset
        self.dataloader = data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, datas in enumerate(self.dataloader):
            yield datas


# unit test
if __name__ == '__main__':
    tf = transforms.Compose([transforms.ToTensor()])
    ds = CustomDataset(r'../dataset/testA', '.jpg', transform=tf)
    loader = CustomDataloader(ds, num_workers=2, shuffle=True)
    for pairs in loader:
        print(pairs['path'])

