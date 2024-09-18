#dropout原理
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

#nn.Dropout以类的形式实现
#nn.functional.dropout以函数的形式实现

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)

torch.nn.functional.dropout(input,p = 0.2, training=True, inplace=False)

#只在训练时使用dropout，在测试时不使用
#训练时drop掉某些神经元，在eval时drop掉一些连接

#在numpy中实现dropout
def train(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1) #第一层
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = layer1 * mask1#实现第一层的dropout
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = layer2 * mask2
    return layer2
def test(x, w1, b1, w2, b2, rate):#测试
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer1 = layer1 * (1 - rate)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    layer2 = layer2 * (1 - rate) #进行矩阵缩放，接近训练阶段
    return layer2

def train2(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1) #第一层
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = layer1 * mask1#实现第一层的dropout
    layer1 = layer1/(1 - rate)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = layer2 * mask2
    layer2 = layer2/(1 - rate)#通过除法避免在测试时计算量过大
    return layer2

def test2(x, w1, b1, w2, b2, rate):#测试
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    return layer2

