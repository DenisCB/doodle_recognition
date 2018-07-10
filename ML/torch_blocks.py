import torch
import torch.nn as nn
import numpy as np


class conv_layer(nn.Module):
    def __init__(self, in_dim, out_ch, kernel, stride, dropout, pool_size):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim[0], out_channels=out_ch,
            kernel_size=kernel, stride=stride)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(pool_size, pool_size), stride=pool_size)
        self.relu = nn.ReLU(inplace=True)
        self.bnorm = nn.BatchNorm2d(num_features=out_ch)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        pic_dim = np.floor((in_dim[1] - (kernel-1))/stride)
        pic_dim = int(np.floor(pic_dim/pool_size))

        self.out_dim = (out_ch, pic_dim, pic_dim)
        self.dropout_value = dropout

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.bnorm(x)
        if self.dropout_value > 0:
            x = self.dropout(x)
        return x


class linear_layer(nn.Module):
    def __init__(self, in_dim, out_dim, activate, BN, dropout):
        super(linear_layer, self).__init__()
        self.lin = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.bnorm = nn.BatchNorm1d(num_features=out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.out_dim = out_dim
        self.activate = activate
        self.BN = BN
        self.dropout_value = dropout

    def forward(self, x):
        x = self.lin(x)
        if self.activate:
            x = self.relu(x)
        if self.BN:
            x = self.bnorm(x)
        if self.dropout_value > 0:
            x = self.dropout(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_dim, out_ch):
        super(InceptionA, self).__init__()

        self.branch5x5_1 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(out_ch, out_ch, kernel_size=(3,1), padding=(1,0))
        self.branch5x5_3 = BasicConv2d(out_ch, out_ch, kernel_size=(1,3), padding=(0,1))
        self.branch5x5_4 = BasicConv2d(out_ch, out_ch, kernel_size=(3,1), padding=(1,0))
        self.branch5x5_5 = BasicConv2d(out_ch, out_ch, kernel_size=(1,3), padding=(0,1))

        self.branch3x3_1 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(out_ch, out_ch, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_3 = BasicConv2d(out_ch, out_ch, kernel_size=(3,1), padding=(1,0))

        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)

        self.branch1x1 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)

        self.out_dim = (out_ch*4, in_dim[1], in_dim[2])

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
        branch5x5 = self.branch5x5_4(branch5x5)
        branch5x5 = self.branch5x5_5(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        output = [branch1x1, branch5x5, branch3x3, branch_pool]
        output = torch.cat(output, 1)
        return output


class InceptionC(nn.Module):

    def __init__(self, in_dim, out_ch):
        super(InceptionC, self).__init__()

        self.branch5x5_1 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(out_ch, out_ch, kernel_size=(5,1), stride=(2,1), padding=(2,0))
        self.branch5x5_3 = BasicConv2d(out_ch, out_ch, kernel_size=(1,5), stride=(1,2), padding=(0,2))

        self.branch3x3_1 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(out_ch, out_ch, kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.branch3x3_3 = BasicConv2d(out_ch, out_ch, kernel_size=(3,1), stride=(2,1), padding=(1,0))

        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch_pool_2 = BasicConv2d(in_dim[0], out_ch, kernel_size=1)

        out_dim_1 = int(np.ceil(in_dim[1]/2))
        out_dim_2 = int(np.ceil(in_dim[2]/2))
        self.out_dim = (out_ch*3, out_dim_1, out_dim_2)

    def forward(self, x):

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        output = [branch5x5, branch3x3, branch_pool]
        output = torch.cat(output, 1)
        return output
