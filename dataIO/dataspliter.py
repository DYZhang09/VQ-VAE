import random
from dataIO.utils import get_img_paths


##################################################
# This file defines the dataspliter which splits #
# the dataset into train_set, val_set and        #
# test_set                                       #
##################################################


class DataSpliter(object):
    """
    dataset spliter that splits the dataset
    """

    def __init__(self,
                 root_path,
                 img_suffix,
                 max_total_num=float("inf"),
                 shuffle=False,
                 train_ratio=0.7,
                 val_ratio=None):
        self.img_paths = get_img_paths(root_path, img_suffix, max_total_num)
        if shuffle:
            random.shuffle(self.img_paths)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.train_num = int(len(self.img_paths) * self.train_ratio)
        self.val_num = self.train_num + int(len(self.img_paths) * self.val_ratio) if self.val_ratio else None

    def train_set(self):
        return self.img_paths[:self.train_num]

    def val_set(self):
        if self.val_ratio is not None:
            return self.img_paths[self.train_num:self.val_num]
        else:
            return None

    def test_set(self):
        if self.val_num:
            return self.img_paths[self.val_num:]
        else:
            return self.img_paths[self.train_num:]
