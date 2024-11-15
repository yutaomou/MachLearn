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

#stride控制卷积核移动步长
#padding控制卷积核边缘填充
#dilation设置一些点为空洞，减少计算量，扩大范围
#groups分组进行卷积运算
#depth_wise不考虑通道间的融合，point_wise不考虑局部融合
#因此，deep-wise可以用于空间混合，point_wise可以用于通道混合

#使用Vision Transformer(VIT) 进行图像分类，Transformer在CV领域前景广阔

#但Conv-mixer（patches + conv）模型性能也很优秀,Conv-mixer完全使用卷积操作
#GELU 激活函数

#实现convmixer
def ConvMixer(h, depth, kernel_size=9, patch_size=7, n_classes=1000):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))# ActBn函数让x经过GELU和归一化之后输出
    Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x}) #定义了一个名为Residual的新类, 接受输入x，并将其传递给类中的第一个组件进行处理，然后将处理后的结果与原始输入x相加
    return Seq(ActBn(nn.Conv2d(3, h, patch_size, stride=patch_size)),
               *[Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))), #*可以解包将循环中的元素作为单独的参数传递给函数和类，实现depth_wise层
                ActBn(nn.Conv2d(h, h, 1)))for i in range(depth)], #实现point_wise层
               nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(h, n_classes)) #自适应平均值，展平，最后映射到n_class的维度，得到分类


#常规卷积
conv_general = nn.Conv2d(3,3,3,padding="same")

#分离卷积
subsconv_space_mixing = nn.Conv2d(3,3,3,groups=3, padding="same") #groups设置为和ingroups一致，使每个通道单独考虑，避免产生通道融合
subconv_channel_mixing = nn.Conv2d(3,3,1) #point_wise卷积

#检查参数数量是否减少
for p in conv_general.parameters():
    print(torch.numel(p))
for p in subsconv_space_mixing.parameters():
    print(torch.numel(p))
for p in subconv_channel_mixing.parameters():
    print(torch.numel(p))

#参数数量减少一半以上，可以降低计算量