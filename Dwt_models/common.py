import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) + dilation - 1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


# def shuffle_channel()

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def pixel_down_shuffle(x, downsacale_factor):
    batchsize, num_channels, height, width = x.size()

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width)


def sp_init(x):
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init(x):
    """
    x = (B, C, H, W)
    x1:偶数行偶数列 x2:奇数行偶数列 x3:偶数行奇数列 x4: 奇数行奇数列
    返回：# (B, 4C, H/2, W/2) 4C:C(LL)-C(HL)-C(LH)-C(HH)
    """
    x01 = x[:, :, 0::2, :] / 2  # (B, C, H, W)-->(B, C, H/2, W)  取偶数行
    x02 = x[:, :, 1::2, :] / 2  # (B, C, H, W)-->(B, C, H/2, W)  取奇数行
    x1 = x01[:, :, :, 0::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取偶数列
    x2 = x02[:, :, :, 0::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取奇数列
    x3 = x01[:, :, :, 1::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取奇数列
    x4 = x02[:, :, :, 1::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取偶数列
    x_LL = x1 + x2 + x3 + x4  # (B, C, H/2, W/2)
    x_HL = -x1 - x2 + x3 + x4  # (B, C, H/2, W/2)
    x_LH = -x1 + x2 - x3 + x4  # (B, C, H/2, W/2)
    x_HH = x1 - x2 - x3 + x4  # (B, C, H/2, W/2)

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)  # (B, 4C, H/2, W/2)


def iwt_init(x):
    """
    x: (B, C, H, W)
    h: (B, C/4, 2H, 2W)
    """
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()  # B, C, H, W
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width  # B, C/4, 2H, 2W
    x1 = x[:, 0:out_channel, :, :] / 2  # 第一个1/4通道   (B, C/4, H, W)
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2  # 第二个1/4通道   (B, C/4, H, W)
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2  # 第三个1/4通道   (B, C/4, H, W)
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2  # 第四个1/4通道   (B, C/4, H, W)

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()  # (B, C/4, 2H, 2W)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4  # 偶数行，偶数列
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4  # 奇数行，偶数列
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4  # 偶数行，奇数列
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4  # 奇数行，奇数列

    return h


class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)


class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)


class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)

class BBlock(nn.Module):
    """
    Conv-bn-act
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=None):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class BBlock1(nn.Module):
    """
    Conv-bn-act
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False):
        super(BBlock1, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        # m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class BBlock2(nn.Module):
    """
    Conv-bn-act
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(BBlock2, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class BBlock_LowF(nn.Module):
    """
    Conv-bn-act
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(BBlock_LowF, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels//4, kernel_size=kernel_size, stride=2, padding=1, bias=bias))      # 一次下采样  112-56
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels//4, out_channels//2, kernel_size=kernel_size, stride=2, padding=1, bias=bias))  # 二次下采样  56-28
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels//2, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=bias))     # 三次下采样  28-14   (BS, 64, 14, 14)
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        m.append(SelectAdaptivePool2d(pool_type='avg'))                                     # (BS, 64, 14, 14)-->(BS, 64, 1, 1)
        m.append(nn.Flatten(1))                                                             # (BS, 64)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class DWT(nn.Module):
    def __init__(self, requires_grad=False):
        super(DWT, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, x):
        return dwt_init(x)

class DWT_spatial_attention3(nn.Module):
    def __init__(self, hidden_d=64):

        super().__init__()
        act = nn.ReLU(True)
        self.dwt = DWT()
        self.conv_l1 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h1 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_l2 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h2 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_l3 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h3 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_sa1 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.conv_sa2 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.conv_sa3 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.sigmoid = nn.Sigmoid()
        self.norm_flag = 0
        self.norm = nn.LayerNorm(hidden_d)


    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$
        return torch.einsum('bihd,bjhd->bijh', query, key)   # query,key: [batch_size, seq_len, d_model]

    def forward(self, x):
        """
        :param x: (2, 3, 224, 224)
        :return: SA1:(2, 112, 112), SA2:(2, 56, 56), SA3:(2, 28, 28)
        """
        # x: raw input (2, 3, 224, 224)
        # 第一次分解
        x_gray = x[:,[0],:,:]                       # (2, 1, 224, 224)
        x_LH_1 = self.dwt(x_gray)                   # (2, 4, 112, 112)
        x_L_1 = self.conv_l1(x_LH_1[:, [0], :, :])  # (2, 1, 112, 112) --> (2, 64)   Q
        x_H_1 = self.conv_h1(x_LH_1[:, 1:, :, :])   # (2, 3, 112, 112) --> (2, 64, 112, 112)   K
        bs, c, h, w = x_H_1.shape

        if self.norm_flag:
            x_L_1 = self.norm(x_L_1.flatten(2).permute(0, 2, 1)).permute(0, 2, 1)  # (B,C,HW)-->(B, HW, C)-->layernorm-->(B,C,HW)
            x_H_1 = self.norm(x_H_1.flatten(2).permute(0, 2, 1))                   # (B,C,HW)-->(B, HW, C)
        else:
            x_L_1 = x_L_1.unsqueeze(2)                              # (B,C,1)
            x_H_1 = x_H_1.flatten(2).permute(0, 2, 1)               # (B,C,HW)-->(B, HW, C)
        correlation1 = torch.bmm(x_H_1,x_L_1)                       # (B,HW,1)                每一行代表每个H和整个L的相关性
        correlation1 = torch.reshape(correlation1,(bs, 1, h, w))    # (B,HW,1) --> (B,1,H,W)
        correlation1 = self.sigmoid(self.conv_sa1(correlation1))
        SA1 = correlation1                                          # (B,H,W) 代表每个位置在高频意义上的重要程度

        # 二次分解
        x_LH_2 = self.dwt(x_LH_1[:, [0], :, :])         # (2, 4, 112, 112)
        x_L_2 = self.conv_l2(x_LH_2[:, [0], :, :])      # (2, 1, 112, 112) --> (2, 64)   Q
        x_H_2 = self.conv_h2(x_LH_2[:, 1:, :, :])       # (2, 3, 112, 112) --> (2, 64, 112, 112)
        bs, c, h, w = x_H_2.shape
        if self.norm_flag:
            x_L_2 = self.norm(x_L_2.flatten(2).permute(0, 2, 1)).permute(0, 2, 1)  # (B,C,HW)-->(B, HW, C)-->layernorm-->(B,C,HW)
            x_H_2 = self.norm(x_H_2.flatten(2).permute(0, 2, 1))      # (B,C,HW)-->(B, HW, C)
        else:
            x_L_2 = x_L_2.unsqueeze(2)                     # (B,HW)-->(B, HW, 1)
            x_H_2 = x_H_2.flatten(2).permute(0, 2, 1)      # (B,C,HW)-->(B, HW, C)
        correlation2 = torch.bmm(x_H_2, x_L_2)             # (B,HW,1) 每一行代表每个H和所有L的相关性
        correlation2 = torch.reshape(correlation2,(bs, 1, h, w))  # 按行求和 (B,HW) --> (B,H,W)
        correlation2 = self.sigmoid(self.conv_sa2(correlation2))
        SA2 = correlation2                                 # (B,H,W) 代表每个位置在高频意义上的重要程度

        # 三次分解
        x_LH_3 = self.dwt(x_LH_2[:, [0], :, :])          # (2, 4, 112, 112)
        x_L_3 = self.conv_l3(x_LH_3[:, [0], :, :])        # (2, 1, 112, 112) --> (2, 64, 112, 112)   Q
        x_H_3 = self.conv_h3(x_LH_3[:, 1:, :, :])       # (2, 3, 112, 112) --> (2, 64, 112, 112)   K
        bs, c, h, w = x_H_3.shape
        if self.norm_flag:
            x_L_3 = self.norm(x_L_3.flatten(2).permute(0, 2, 1)).permute(0, 2, 1)  # (B,C,HW)-->(B, HW, C)-->layernorm-->(B,C,HW)
            x_H_3 = self.norm(x_H_3.flatten(2).permute(0, 2, 1))      # (B,C,HW)-->(B, HW, C)
        else:
            x_L_3 = x_L_3.unsqueeze(2)                         # (B,HW)-->(B, HW, 1)
            x_H_3 = x_H_3.flatten(2).permute(0, 2, 1)          # (B,C,HW)-->(B, HW, C)
        correlation3 = torch.bmm(x_H_3, x_L_3)                 # (B,HW,1) 每一行代表每个H和所有L的相关性
        correlation3 = torch.reshape(correlation3,(bs, 1, h, w))  # 按行求和 (B,HW) --> (B,H,W)
        correlation3 = self.sigmoid(self.conv_sa3(correlation3))
        SA3 = correlation3                                     # (B,H,W) 代表每个位置在高频意义上的重要程度

        return SA1,SA2,SA3


class IWT(nn.Module):
    def __init__(self, requires_grad=False):
        super(IWT, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.create_graph = False
            self.volatile = True


class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.volatile = True


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class DBlock_com(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com1(nn.Module):
    """
    conv(dilation=2)-bn-act--conv-bn-act 总的输入输出，即in_channels, out_channels参数必须要相同
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv1(nn.Module):
    """
    conv(d=2)-bn-act--conv-bn-act
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com2(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv2(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class ShuffleBlock(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, conv_groups=1):
        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
            self, conv, conv1, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Block(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




