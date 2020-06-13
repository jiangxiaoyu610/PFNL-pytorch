import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms

from PIL import Image
from scipy.ndimage.filters import gaussian_filter


random.seed(10)
np.random.seed(10)

PIL_TRANS = torchvision.transforms.ToPILImage()
TENSOR_TRANS = torchvision.transforms.ToTensor()


def load_image(file):
    """
    读取给定的图像文件
    :return:
    """
    image = Image.open(file)
    image = image.convert('RGB')

    return image


def get_data_set_list(folder_list_path, shuffle=False):
    """
    读取数据集文件名。
    :param folder_list_path:
    :param shuffle:
    :return:
    """
    folder_list = open(folder_list_path, 'rt').read().splitlines()
    if shuffle:
        random.shuffle(folder_list)

    files_list = []
    for folder in folder_list:
        pattern = os.path.join(folder, 'truth/*.png')
        files = sorted(glob.glob(pattern))
        files_list.append(files)

    return files_list


def get_gaussian_filter(size, sigma):
    """
    生成高斯过滤器。
    :param size: 过滤器大小
    :param sigma: 标准差
    :return:
    """
    template = np.zeros((size, size))
    template[size//2, size//2] = 1
    return gaussian_filter(template, sigma)


def down_sample_with_blur(images, kernel, scale):
    """
    对图片进行高斯模糊和下采样。
    :param images:
    :param kernel:
    :param scale:
    :return:
    """
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = (pad_left, pad_right, pad_top, pad_bottom)

    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(3, 1, 1, 1)

    reflect_layer = torch.nn.ReflectionPad2d(pad_array)

    padding_images = reflect_layer(images)
    # 在 pytorch 中通过设置 groups 参数来实现 depthwise_conv。
    # 详情参考：https://www.jianshu.com/p/20ba3d8f283c
    output = torch.nn.functional.conv2d(padding_images, kernel, stride=scale, groups=3)

    return output
