import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms


##################################################
# This file defines some util functions used to  #
# build the dataset and dataloader               #
##################################################


def get_img_paths(root, suffix='.png', max_size=float("inf")):
    assert os.path.isdir(root), "the root is not a directory"

    img_paths = []
    len = 0
    for root, _, filenames in sorted(os.walk(root)):
        for filename in filenames:
            if filename.endswith(suffix):
                img_paths.append(os.path.join(root, filename))
                len += 1
                if len > max_size:
                    return img_paths
    return img_paths


def read_image(path, read_as_numpy=False):
    img = Image.open(path).convert('RGB')
    if read_as_numpy:
        return np.array(img)
    return img


def write_image(path, image, is_numpy=False):
    if is_numpy:
        image = Image.fromarray(image)
    image.save(path)


def tensor2img_transform():
    tf = [transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
          transforms.ToPILImage()]
    tf = transforms.Compose(tf)
    return tf


def img2tensor_transform():
    tf = [transforms.ToTensor(),
          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    tf = transforms.Compose(tf)
    return tf


def tensor2img(img_tensor):
    tf = tensor2img_transform()
    return tf(img_tensor).convert('RGB')


def img2tensor(img):
    tf = img2tensor_transform()
    return tf(img)


# unit test
if __name__ == '__main__':
    img_paths = get_img_paths(r'../dataset', suffix='.jpg')
    print(img_paths)
    for path in img_paths:
        img = read_image(path, read_as_numpy=True)
        print(img.shape)
        img_tensor = img2tensor(img)
        out = tensor2img(img_tensor)
        write_image(r'./test.jpg', out)
