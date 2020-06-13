import glob

from PIL import Image
from local_common import *
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """
    自定义数据集类。
    根据给定的图像路径文件，读取相应的图像文件。
    经过对输入输出的仔细对比，确定可以与原 PFNL 代码达到同等效果。
    """
    def __init__(self, args, repeat=None):
        """
        初始化
        :param args:
        :param repeat:
        """
        self.repeat = repeat
        self.max_data_set_length = int(args.batch_size * args.max_step)

        self.size = args.ground_truth_size  # size: ground truth 图像的长宽
        self.scale = args.scale  # scale: 图像下采样的倍数
        self.n_frames = args.num_frames  # n_frames: 设置几个帧为一个样本
        self.files_list = get_data_set_list(args.train_files_list, shuffle=True)  # folder_list_path: 数据集文件
        self.length = len(self.files_list)

        return

    def __getitem__(self, index):
        """
        一次读取 n_frames 个图像，作为一个样本。
        :param index:
        :return:
        """
        index = index % self.length
        files = self.files_list[index]
        n_files = len(files)

        begin = random.randint(0, n_files-self.n_frames)
        end = begin + self.n_frames
        selected_files = files[begin: end]

        images = [load_image(file) for file in selected_files]
        input_images, ground_truth = self.pre_process(images)

        return input_images, ground_truth

    def __len__(self):
        if self.repeat is None:
            return self.max_data_set_length
        else:
            return int(self.length * self.repeat)

    def pre_process(self, images):
        """
        对图像进行预处理。
        主要步骤如下：
            1）随机剪裁
            2）像素归一化
            3）随机进行：上下翻转、左右翻转、90度翻转
            4）高斯模糊过滤同时对原图像进行 self.scale 倍下采样
            5）只取中间一帧作为 Ground Truth
        :param images:
        :return:
        """
        kernel = get_gaussian_filter(13, 1.6)  # 13 and 1.6 for x4 down sample
        # 随机生成剪裁区域
        width, height = images[0].size[0], images[0].size[1]
        left, top = random.randint(0, width-self.size), random.randint(0, height-self.size)
        right, bottom = left+self.size, top+self.size
        crop_array = [left, top, right, bottom]

        random_flip = np.random.uniform(0, 1, (3,))
        for i in range(len(images)):
            # 剪裁
            images[i] = images[i].crop(crop_array)

            # 随机进行：上下翻转、左右翻转、90度翻转
            if random_flip[0] >= 0.5:
                images[i] = images[i].transpose(Image.FLIP_TOP_BOTTOM)
            if random_flip[1] >= 0.5:
                images[i] = images[i].transpose(Image.FLIP_LEFT_RIGHT)
            if random_flip[2] >= 0.5:
                images[i] = images[i].transpose(Image.TRANSPOSE)

            # 像素归一化
            images[i] = torch.Tensor(np.array(images[i])) / 255
            images[i] = np.transpose(images[i], (2, 0, 1))

        # 高斯模糊和下采样
        images = torch.stack(images, 0)
        input_images = down_sample_with_blur(images, kernel, self.scale)
        ground_truth = images[self.n_frames//2: self.n_frames//2+1]

        return input_images, ground_truth
