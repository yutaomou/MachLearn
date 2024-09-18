from collections import OrderedDict

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

#embedding
#分布式表征
#用向量代表单词
#输入onehot变量，经过embedding table 变为长度固定的embedding vector
n, d, m = 3, 5, 7
embedding = nn.Embedding(n, d, max_norm=1)
Weight = torch.randn((m, d), requires_grad=True)
idx = torch.tensor([1, 2])
a = embedding.weight.clone() @ Weight.t()
#微分
b = embedding(idx) @ Weight.t() #建立全连接网络,输入为index，权重为embeddingtable 输出为每个单词的embedding vector
#通过不断训练embedding table来达到最优

#自动微分中的forward和Reverse模式

#forward mode
def f(x):
    v = x
    for i in range(3):
        v = v * 4 *(1 - v)
    return v

def different_f(x):
    (v, dv) = (x, 1)
    for i in range(3):
        (v, dv) = (v * 4 * (1 - v), 4 * dv - 8*v*dv)
    return (v, dv)

#reverse mode具有梯度累计，forward mode没有

#forward mode for AD
#|y| * |b|(|b|*|a|,|a|*|x|) = bax + ybx
#reverse mode for AD
#|y|*|b| |b|*|a| |a|*|x| = yba +yax
#当x>y，即输入特征大于输出特征时，reverse mode计算量较小
#当x<y，即输入特征小于输出特征时，forward mode计算量较小
#在机器学习中，输出特征就是loss值，所以输入特征较多
#但是可以通过统计每一层的输入，输出特征比，来确定每一层采用哪种mode
