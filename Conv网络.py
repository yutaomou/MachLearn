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
import math
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#如果输入不为1个通道，把output的多个矩阵进行点积求和，得到最终的output
#kernel进行Z字型滑动
#内积越大，两个向量越相似

#调用pytorch中的卷积网络
in_channels = 1
out_channels = 1
kernel_size = 3 #如果为标量则代表是正方形，如果是元组，则代表为矩形
bias = False
batch_size = 1
input_size = [batch_size, in_channels, 4, 4] #4,4代表输入的图片大小
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")  #输入通道数，输出通道数，卷积核大小，填充,定义一个卷积层
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
print(input_feature_map)
print(output_feature_map)
print(conv_layer.weight) #打印卷积层的kernel out_channels*in_channels*kernel_size

#同样，用functional中的conv实现
#需要手动制定weight和bias
output_feature_map1 = F.conv2d(input_feature_map, conv_layer.weight, bias=conv_layer.bias)
