from local_common import *
from nonlocal_block import NonLocalBlock
from torch.nn import init, Conv2d, Sequential


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)  # (n, bs, bs, c//bs^2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (n, c//bs^2, h, bs, w, bs)
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)  # (n, c//bs^2, h * bs, w * bs)
        return x


class SpaceToDepth(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)  # (n, c, h//bs, bs, w//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (n, bs, bs, c, h//bs, w//bs)
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)  # (n, c*bs^2, h//bs, w//bs)
        return x


class PFNL(torch.nn.Module):
    def __init__(self, args):
        """
        模型参数、神经网路层初始化
        :param args: 每个样本包含的帧数
        """
        super(PFNL, self).__init__()
        self.args = args
        self.n_frames = args.num_frames
        self.train_size = (args.in_size, args.in_size)
        self.eval_size = args.eval_in_size

        # 在输入为正方矩阵的简单情况下，padding = (k_size-1)/2 时为 SAME
        self.convolution_layer0 = Sequential(Conv2d(3, args.n_filters, 5, args.strides, 2), args.activate())
        init.xavier_uniform_(self.convolution_layer0[0].weight)

        self.convolution_layer1 = torch.nn.ModuleList([
            Sequential(
                Conv2d(args.n_filters, args.n_filters, args.k_size, args.strides, (args.k_size-1) // 2),
                args.activate()
            )
            for _ in range(args.n_block)])

        self.convolution_layer10 = torch.nn.ModuleList([
            Sequential(
                Conv2d(self.n_frames * args.n_filters, args.n_filters, 1, args.strides, 0),
                args.activate()
            )
            for _ in range(args.n_block)])

        self.convolution_layer2 = torch.nn.ModuleList([
            Sequential(
                Conv2d(2 * args.n_filters, args.n_filters, args.k_size, args.strides, (args.k_size-1) // 2),
                args.activate()
            )
            for _ in range(args.n_block)])

        # xavier初始化参数
        for i in range(args.n_block):
            init.xavier_uniform_(self.convolution_layer1[i][0].weight)
            init.xavier_uniform_(self.convolution_layer10[i][0].weight)
            init.xavier_uniform_(self.convolution_layer2[i][0].weight)

        self.convolution_merge_layer1 = Sequential(
            Conv2d(self.n_frames * args.n_filters, 48, 3, args.strides, 1), args.activate()
        )
        init.xavier_uniform_(self.convolution_merge_layer1[0].weight)

        self.convolution_merge_layer2 = Sequential(
            Conv2d(48 // (2 * 2), 12, 3, args.strides, 1), args.activate()
        )
        init.xavier_uniform_(self.convolution_merge_layer2[0].weight)

        # 参考：https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        self.space_to_depth, self.depth_to_space = SpaceToDepth(2), DepthToSpace(2)
        self.nonlocal_block = NonLocalBlock(2*2*21, 3*self.n_frames*4, 1, 1)

    def forward(self, input_image):
        # 注意！输入图片的 shape 应该变为 batch_size * n_frames * channel * width * height
        input0 = [input_image[:, i, :, :, :] for i in range(self.n_frames)]
        input0 = torch.cat(input0, 1)

        input1 = self.space_to_depth(input0)
        input1 = self.nonlocal_block(input1)
        input1 = self.depth_to_space(input1)
        input0 += input1

        input0 = torch.split(input0, 3, dim=1)
        input0 = [self.convolution_layer0(frame) for frame in input0]

        basic = input_image[:, self.n_frames//2, :, :, :].squeeze(0)
        basic = self.perform_bicubic(basic, self.args.scale)
        basic = basic.unsqueeze(0)

        for i in range(self.args.n_block):
            input1 = [self.convolution_layer1[i](frame) for frame in input0]
            base = torch.cat(input1, 1)
            base = self.convolution_layer10[i](base)

            input2 = [torch.cat([base, frame], 1) for frame in input1]
            input2 = [self.convolution_layer2[i](frame) for frame in input2]
            input0 = [torch.add(input0[j], input2[j]) for j in range(self.n_frames)]

        merge = torch.cat(input0, 1)
        merge = self.convolution_merge_layer1(merge)

        large = self.depth_to_space(merge)
        output = self.convolution_merge_layer2(large)
        output = self.depth_to_space(output)

        return torch.stack([output+basic], 1)

    @staticmethod
    def perform_bicubic(image_tensor, scale):
        """
        对 tensor 类型的图像进行双三次线性差值。
        :param image_tensor:
        :param scale: 放大倍数
        :return:
        """
        c, h, w = image_tensor.shape

        image = PIL_TRANS(image_tensor)  # 注意在图片中会变成 w * h
        image = image.resize((w * scale, h * scale), Image.BICUBIC)
        output = TENSOR_TRANS(image)

        return output