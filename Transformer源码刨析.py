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

#Seq2Seq一般由encoder,attention,decoder组成
#Seq2Seq可以由CNN(卷积神经网络)，RNN(循环神经网络)，Transformer构建



"""
                  |--平移不变形
      |--权重共享--|
      |           |--可并行运算
CNN---|--滑动窗口  局部关联性建模  靠多层堆叠实现长程建模
      |
      |--相对位置敏感(打乱输入的顺序，输出改变)，绝对位置不敏感(将输入进行旋转，卷积结果相同)
"""


"""
                    |--对顺序敏感
                    |--串行计算耗时
RNN--依次有序递归建模--|--长程建模能力弱(不能跳过)
     (必须按顺序计算)  |--计算复杂度与序列长度呈线性关系
                    |--单步计算复杂度不变
                    |--对相对位置敏感，绝对位置也敏感
"""

"""
                                             |--可并行运算
               |--无局部假设(并不对局部进行建模)--|
              |                              |--对相对位置不敏感
              |
              |                      |--需要位置编码来反应位置变化对于特征的影响
Transformer--| --无有序假设(与RNN不同)--|
              |                      |--对绝对位置不敏感
               |
               |                     |--擅长长短程建模
               |--任意两字符都可以建模--|
                                     |--自注意力机制需要序列长度的平方级别复杂度

"""


#实现自注意力机制
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) #在mask为0的地方填充负无穷
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

