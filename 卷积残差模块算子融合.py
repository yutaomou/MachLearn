#卷积残差模块算子融合
import torch
import time
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

#每个通道都会有两套3*3的矩阵（因为输入有两个通道），对输入的两个通道进行滑动卷积，然后将两个输入通道的滑动卷积结果相加再加bias，就得到结果
#point-wise convolution一般是1*1的一个卷积，打破了卷积的假设（局部关联性和平移不变性（旋转图片会得到一样的值））不考虑局部关联性，只考虑自身
#deep-wise convolution一般都是3*3的卷积，将groups设置为大于1的数，即把卷积分成几部分，可以降低计算量

#conv_residual_block_fusion

#对于res_block模型
#公式：res_block = 3*3 conv + 1*1 conv +input
in_channels = 2
ou_channels = 2
kernel_size = 3
w = 9
h = 9

t0 = time.time()
#原生写法
#一般卷积输入的向量维度为（batch_size*in_channels*width*height）
x = torch.ones(1, in_channels, w, h) #输入张量大小
conv_2d = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
conv_2d_pointwise = nn.Conv2d(in_channels, ou_channels, 1)
result_1 = conv_2d(x) + conv_2d_pointwise(x) + x #原生写法的输出
print(result_1)

t1 = time.time()
#算子融合写法
#把point_wise卷积和x本身都写成3*3的卷积
#最终把三个卷积都写成一个卷积
#F.pad() 是pytorch 内置的 tensor 扩充函数，便于对数据集图像或中间层特征进行维度扩充

#把pointwise卷积写成3*3的卷积，但是依然不关联相邻点关联性和通道关联性

#Step1 改造
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, (1, 1, 1, 1, 0, 0, 0, 0))#在上面一行，下面一行，左边一列，右边一列都填充0，即把2*2*1*1->2*2*3*3
conv_2d_for_pointwise = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
conv_2d_for_pointwise.weight = torch.nn.Parameter(pointwise_to_conv_weight) #修改weight
conv_2d_for_pointwise.bias = torch.nn.Parameter(conv_2d_pointwise.bias)  #修改bias

#把x本身也写成3*3的卷积
zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0) #构建全0矩阵,增加维度为3维
stars = torch.unsqueeze(F.pad(torch.ones(1, 1), [1, 1, 1, 1]), 0) #构建全1矩阵，填充1，增加维度为3维
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], dim=0), 0) #第一个输出通道的卷积核,是一个4维的张量
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], dim=0), 0) #第二个输出通道的卷积核,是一个4维的张量
identity_conv_weight = torch.cat([stars_zeros, zeros_stars], dim=0) #把两个卷积核拼接成一个4维张量
identity_conv_bias = torch.zeros([ou_channels]) #设置bias为0
conv_2d_for_identity = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
conv_2d_for_identity.weight = torch.nn.Parameter(identity_conv_weight)
conv_2d_for_identity.bias = torch.nn.Parameter(identity_conv_bias)

result_2 = conv_2d(x) + conv_2d_pointwise(x) + conv_2d_for_identity(x)

print(result_2)

print(torch.all(torch.isclose(result_1, result_2))) #判断两个张量是否相同

#Step2 融合
conv_2d_for_fusion = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
conv_2d_for_fusion.weight = torch.nn.Parameter(conv_2d.weight.data + conv_2d_for_pointwise.weight.data + conv_2d_for_identity.weight.data) #构成融合卷积的weight
conv_2d_for_fusion.bias = torch.nn.Parameter(conv_2d.bias.data + conv_2d_for_pointwise.bias.data + conv_2d_for_identity.bias.data) #构成融合卷积的bias
result_3 = conv_2d_for_fusion(x) #融合卷积的输出

print(torch.all(torch.isclose(result_3, result_2))) #判断融合卷积和上面两个卷积的输出是否相同

t2 = time.time()

print("原生写法耗时", t1 - t0)
print("融合写法耗时", t2 -t1)
#算子融合可以使三个卷积合并成一个卷积