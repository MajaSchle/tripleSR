import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MLP(nn.Module):
    def __init__(self, in_dim=128, out_dim=1, depth=4, width=256):
        super().__init__()
        stage_one_layers = []
        for i in range(depth):
            if i == 0:
                stage_one_layers.append(nn.Linear(in_dim, width))
                stage_one_layers.append(nn.ReLU())
            elif i == depth - 1:
                stage_one_layers.append(nn.Linear(width, in_dim))
                stage_one_layers.append(nn.ReLU())
            else:
                stage_one_layers.append(nn.Linear(width, width))
                stage_one_layers.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one_layers)

        stage_two_layers = []
        for i in range(depth):
            if i == 0:
                stage_two_layers.append(nn.Linear(in_dim, width))
                stage_two_layers.append(nn.ReLU())
            elif i == depth - 1:
                stage_two_layers.append(nn.Linear(width, out_dim))
                stage_two_layers.append(nn.ReLU())
            else:
                stage_two_layers.append(nn.Linear(width, width))
                stage_two_layers.append(nn.ReLU())
        self.stage_two = nn.Sequential(*stage_two_layers)

    def forward(self, x):
        h = self.stage_one(x)
        x_res = x + h
        out = self.stage_two(x_res)
        return out


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='trilinear', scale_factor=1),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)


        self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)

        x = F.relu(self.conv1(from_up))
        x = F.relu(self.conv2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim=128, out_dim=1, in_channels=3, depth=5,
                 start_filts=128):
        super(Decoder, self).__init__()
        self.up_mode = 'upsample'
        self.merge_mode = 'concat'
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth


        self.up_convs = []
        outs = self.start_filts
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=self.up_mode,
                merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, 1)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        for i, module in enumerate(self.up_convs):
            x = module(x)

        x = self.conv_final(x)
        return x
