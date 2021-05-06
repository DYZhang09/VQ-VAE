import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from model.vqvae import VQVAE


###########################################################
# This file defines the trainer used for training models  #
###########################################################


class Trainer(object):
    """
    model trainer
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 train_dataloader,
                 test_dataloader,
                 log_dir,
                 epochs,
                 log_per_epoch=10,
                 use_gpu=False,
                 device=torch.device('cuda')):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.log_dir = log_dir
        self.epochs = epochs
        self.log_per_epoch = log_per_epoch
        self.use_gpu = use_gpu
        self.device = device

    def __eval(self):
        if self.use_gpu:
            self.model.to(self.device)

        for i, data in enumerate(self.test_loader):
            data = data[0]
            if self.use_gpu:
                data = data.to(self.device)
            print("|eval loss: %.6f" % self.model.evaluate(data))

    def train(self):
        if self.use_gpu:
            self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            for i, data in enumerate(self.train_loader):
                data = data[0]
                if self.use_gpu:
                    data = data.to(self.device)
                self.model(data)
                loss = self.model.calc_loss()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("|epoch: %d|loss: %.6f|" % (epoch, loss))
            self.__eval()
            if epoch % self.log_per_epoch == 0 and self.log_dir is not None:
                path = os.path.join(self.log_dir, 'epoch_%d_vqvae_weight.pth' % epoch)
                torch.save(self.model.state_dict(), path)
