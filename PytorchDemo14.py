#卷积残差模块算子融合
import torch
import os
import numpy as np
from mpmath import eps
from torch import nn, optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.io import read_image
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#对于每一个训练样本，R_drop 会对这些模型的KL散度进行一个最小化
#KL散度（Kullback-Leibler Divergence）是一种用来衡量两个概率分布之间差异性的统计量。KL散度可以看作是从一个概率分布到另一个概率分布的信息增益或者说信息损耗。它是非对称的，这意味着从分布P到分布Q的KL散度与从Q到P的KL散度通常是不同的。


#二维卷积API
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',device=None,dtype=None)
#in_channels:输入通道数
#out_channels:输出通道数
#kernel_size:卷积核大小
#stride:步长（滑动窗口的长度）
#padding:填充
#dilation:空洞，可以增大卷积的一个范围，扩大局部关联
#groups:分组卷积，默认为1，表示普通卷积，当groups=in_channels（且大于1）时，为深度卷积，当groups=out_channels时，为分组卷积
#bias:是否使用偏置

#函数形式的卷积：torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
def func(x):
    return x

label = torch.tensor([1, 0, 1, 0])
def kl(logits1, logits2):
    return torch.mean(F.kl_div(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1), reduction='batchmean'))
class Conv2d(nn.Module):   #实现r_dropout
    def train_r_dropout(rate, x, w1, b1, w2, b2, nll=None):
        x = torch.cat([x, x], 0)#拼接，使batch_size扩大两倍
        layer1 = np.maximum(0, np.dot(w1, x) + b1)  # 第一层
        mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
        layer1 = layer1 * mask1  # 实现第一层的dropout
        layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
        mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
        layer2 = layer2 * mask2

        logits = func(layer2)
        logits1, logits2 = logits[:len(logits) // 2], logits[len(logits) // 2:]#将logits列表或张量从头到中间位置切分为logits1，从中间到末尾切分为logits2
        nll1 = nll(logits1,label)#计算第一个logits的nll损失(衡量模型预测值与实际标签之间的差异)
        nll2 = nll(logits2,label)#计算第二个logits的nll损失
        kl_loss = kl(logits1, logits2)
        loss = nll1 + nll2 + kl_loss # KL散度
        return loss


#实现Conv
conv_layer = torch.nn.Conv2d(2, 2, 3, padding="same")#输入通道数，输出通道数，卷积核大小，填充,定义一个卷积层
for i in conv_layer.named_parameters():
    print(i)

print(conv_layer.weight)#得到权重
#权重的维度与输出通道的数量一致，即（输出通道*输入通道*卷积核大小）
#conv_layer.bias.size()=输出通道大小


