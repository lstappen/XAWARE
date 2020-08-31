import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class FABlock(nn.Module):
    def __init__(self, channel):
        super(FABlock, self).__init__()
        self.fa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fa(x)
        return x * y


class GABlock(nn.Module):
    def __init__(self, channel):
        super(GABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ga = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ga(y)
        return x * y


class Stack(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Stack, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.PReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.galayer = GABlock(dim) 
        self.falayer = FABlock(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.galayer(res)
        res = self.falayer(res)
        res = res + x
        res += x
        return res


class Group(nn.Module):
    # create multiple GA FA stacks in one block
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Stack(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


if __name__ == "__main__":
    print('test')
