from dataIO.dataset import CustomDataset
from dataIO.dataloader import CustomDataloader
from dataIO.dataspliter import DataSpliter


##########################################################
# This file defines the data provider that provides data #
# for training, validation and test                      #
##########################################################


class DataProvider(object):
    """
    data provider
    """

    def __init__(self,
                 dataset_root,
                 img_suffix,
                 max_total_num=float("inf"),
                 train_ratio=0.7,
                 val_ratio=None,
                 shuffle=False,
                 transform=None,
                 batch_size=1,
                 num_workers=1):
        self.spliter = DataSpliter(dataset_root, img_suffix, max_total_num, shuffle, train_ratio, val_ratio)
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_loader(self):
        return CustomDataloader(CustomDataset(self.spliter.train_set(), self.transform),
                                self.batch_size,
                                num_workers=self.num_workers)

    def val_loader(self):
        return CustomDataloader(CustomDataset(self.spliter.val_set(), self.transform),
                                self.batch_size,
                                num_workers=self.num_workers)

    def test_loader(self):
        return CustomDataloader(CustomDataset(self.spliter.test_set(), self.transform),
                                self.batch_size,
                                num_workers=self.num_workers)


# unit test:
if __name__ == '__main__':
    provider = DataProvider(r'../dataset/', '.jpg', batch_size=2, num_workers=2)
    train_loader = provider.train_loader()
    test_loader = provider.test_loader()
    print(len(train_loader), len(test_loader))
    for data in test_loader:
        print(data[0].shape)