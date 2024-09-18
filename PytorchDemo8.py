#nn.Sequential and ModuleList
from collections import OrderedDict

import torch
import os
import numpy as np
from torch import nn, optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#train函数,model.train设置模型为训练模式
#dropout函数可以随机丢弃一些神经元，防止过拟合，继承自module类
#eval函数使模型进入推理模式
#只需要对最上级的模型调用train/eval函数，其他子模块会自动继承
#requires_grad_ 设置参数是否需要计算梯度
#zero_grad 清空梯度,在训练开始前对优化器进行调用
#__repr__函数可以打印模型信息，属于魔法函数
#dir(model)可以列出模型中的所有属性

#以下均为Module的子类

#nn.container已经过时

#Sequential类决定了进行前行计算的顺序

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Conv2d(3, 4, 3),
    nn.ReLU(),
)

#也可以使用有序字典
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2, 3)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(3, 4)),
    ('relu2', nn.ReLU()),
]))

print(model._modules)

#ModuleList类可以把许多子module放到一个列表中，继承自Module
#ParameterList类可以把许多参数放到一个列表中
#ParameterDict类可以把许多参数放到一个字典中

