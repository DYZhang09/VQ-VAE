import torch
import os
import argparse

from dataIO.dataprovider import DataProvider
from dataIO.utils import img2tensor_transform
from model.vqvae import VQVAE
from trainer.trainer import Trainer

########################################################
# This file defines the training code                  #
########################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train the vqvae(only for stage 1)')
    parser.add_argument('--dataroot', required=True, help='the root path of dataset')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='the suffix of image files')
    parser.add_argument('--max_img_num', type=int, default=float("inf"), help='the max num of images to be loaded')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='the ratio of images used in training')
    parser.add_argument('--batch_size', type=int, default=1, help='the input batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle the dataset')
    parser.add_argument('--embed_size', type=int, default=512, help='the num of vectors in codebook in vqvae')
    parser.add_argument('--embed_dim', type=int, default=128, help='the dim of vectors in codebook in vqvae')
    parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate')
    parser.add_argument('--loss_weight', type=float, default=1, help='the commit loss weight')
    parser.add_argument('--epochs', type=int, default=1, help='the training epochs')
    parser.add_argument('--use_gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--weight_dir', type=str, default='./weights', help='where to store the weight file')
    parser.add_argument('--log_per_epochs', type=int, default=10, help='log the weight of vqvae every ? epochs')
    parser.add_argument('--weight_file', type=str, default=None,
                        help='where the weight file resides if want to resume training')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume training')

    args = parser.parse_args()

    provider = DataProvider(dataset_root=args.dataroot,
                            img_suffix=args.img_suffix,
                            max_total_num=args.max_img_num,
                            train_ratio=args.train_ratio,
                            shuffle=args.shuffle,
                            transform=img2tensor_transform(),
                            batch_size=args.batch_size)
    model = VQVAE(embed_size=args.embed_size,
                  embed_dim=args.embed_dim,
                  commit_loss_weight=args.loss_weight,
                  lr=args.lr)
    if args.resume and args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer = Trainer(model=model,
                      train_dataloader=provider.train_loader(),
                      test_dataloader=provider.test_loader(),
                      epochs=args.epochs,
                      use_gpu=args.use_gpu,
                      log_dir=args.weight_dir,
                      log_per_epoch=args.log_per_epochs)
    trainer.train()
