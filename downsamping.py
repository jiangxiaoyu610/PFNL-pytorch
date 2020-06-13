import os
import torch
import numpy as np
import torchvision

from PIL import Image
from scipy.ndimage.filters import gaussian_filter

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

    print(images.shape)
    reflect_layer = torch.nn.ReflectionPad2d(pad_array)

    padding_images = reflect_layer(images)
    # 在 pytorch 中通过设置 groups 参数来实现 depthwise_conv。
    # 详情参考：https://www.jianshu.com/p/20ba3d8f283c
    output = torch.nn.functional.conv2d(padding_images, kernel, stride=scale, groups=3)

    return output


def process():
    """
    对数据集进行下采样
    :return:
    """
    data_folder = './down_sample'
    kernel = get_gaussian_filter(13, 1.6)

    for root, folders, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            """
            image = load_image(file_path)
            image.resize((image.size[0]//4, image[1]//4))
            image.save(file_path)
            """
            image = load_image(file_path)
            tensor = TENSOR_TRANS(image)
            tensor = tensor.unsqueeze(0)
            tensor = down_sample_with_blur(tensor, kernel, 4)
            tensor = tensor.squeeze(0)
            image = PIL_TRANS(tensor)
            image.save(file_path)

    return


if __name__ == '__main__':
    process()
