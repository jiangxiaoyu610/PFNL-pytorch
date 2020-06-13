from local_common import *


class NonLocalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, sub_sample=1, nltype=0):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub_sample = sub_sample
        self.nltype = nltype

        self.convolution_g = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_g = torch.nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_phi = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_phi = torch.nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_theta = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.convolution_y = torch.nn.Conv2d(self.out_channels, self.in_channels, 1, 1, 0)

        self.relu = torch.nn.ReLU()

    def forward(self, input_x):
        batch_size, in_channels, height, width = input_x.shape

        assert self.nltype <= 2, ValueError("nltype must <= 2")
        # g
        g = self.convolution_g(input_x)
        if self.sub_sample > 1:
            g = self.pooling_g(g)
        # phi
        if self.nltype == 0 or self.nltype == 2:
            phi = self.convolution_phi(input_x)
        elif self.nltype == 1:
            phi = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))
        if self.sub_sample > 1:
            phi = self.pooling_phi(phi)

        # theta
        if self.nltype == 0 or self.nltype == 2:
            theta = self.convolution_theta(input_x)
        elif self.nltype == 1:
            theta = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        g_x = g.reshape([batch_size, -1, self.out_channels])
        theta_x = theta.reshape([batch_size, -1, self.out_channels])
        phi_x = phi.reshape([batch_size, -1, self.out_channels])
        phi_x = phi_x.permute(0, 2, 1)

        f = np.matmul(theta_x, phi_x)
        if self.nltype <= 1:
            f = torch.exp(f)
            f_softmax = f / f.sum(dim=-1, keepdim=True)
        elif self.nltype == 2:
            self.relu(f)
            f_mean = f.sum(dim=2, keepdim=True)
            f_softmax = f / f_mean
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        y = torch.matmul(f_softmax, g_x)
        y = y.reshape([batch_size, self.out_channels, height, width])
        z = self.convolution_y(y)

        return z
