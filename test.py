import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model.vqvae import VQVAE
from dataIO.dataprovider import DataProvider
from dataIO.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test the vqvae model')
    parser.add_argument('--dataroot', type=str, default='./dataset', help='the root path of dataset')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='the suffix of image files')
    parser.add_argument('--max_img_num', type=int, default=float("inf"), help='the max num of images to be loaded')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='the ratio of images used to test')
    parser.add_argument('--batch_size', type=int, default=1, help='the input batch size')
    parser.add_argument('--use_gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--weight_file', type=str, default='./weights/epoch_10_vqvae_weight.pth',
                        help='where the weight file resides if want to resume training')
    parser.add_argument('--device', type=str, default='0', help='the cuda device')

    args = parser.parse_args()

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda')

    provider = DataProvider(dataset_root=args.dataroot,
                            img_suffix=args.img_suffix,
                            max_total_num=args.max_img_num,
                            train_ratio=1 - args.test_ratio,
                            transform=img2tensor_transform(),
                            batch_size=args.batch_size)
    model = VQVAE()
    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))
    if args.use_gpu:
        model = model.to(device)

    dataloader = provider.test_loader()
    for data in dataloader:
        img = data['img']
        path = data['path']
        if args.use_gpu:
            img = img.to(device)
        model.set_input(img)
        out = model.test()
        N = out.shape[0]
        for i in range(N):
            out_img = tensor2img(out[i])
            write_image(os.path.join(r'./out',
                                     os.path.basename(path[i])),
                        out_img)
