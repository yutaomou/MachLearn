#nn.Module中的方法
import torch
import os
import numpy as np
from torch import nn
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x)) # 激活函数
        return F.relu(self.linear2(x))

#add_module方法添加子模块

#apply方法

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0) #  对weight函数进行填充
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2),nn.Linear(2, 2))
net.apply(init_weights) # 对net进行初始化


#bfloat16 类型转换

#buffers 方法
#buffer 缓冲区是用来存储那些不通过反向传播更新的数据
#buffers方法返回一个迭代器，可以遍历所有的buffer
#parameters这些是模型的主要参数，通常是在训练过程中需要通过反向传播来更新的权重和偏置

for buf in Model.buffers():
    print(type(buf), buf.size())

#children()返回所有的子模块

#cpu()将所有的模型和参数搬到cpu上

#eval()方法设置为评估模式

#get_parameters()方法可以获取所有的参数

#load_state_dict()方法可以加载所有的模型参数和buffer
#torch.save()方法可以保存模型以字典的格式存储
optimizer = torch.optim.SGD(Model.parameters(), lr=0.001) # 设置学习率
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4
torch.save({
    'epoch': EPOCH, # 训练的轮数
    'model_state_dict': net.state_dict(), # 模型的参数和buffer
    'optimizer_state_dict': optimizer.state_dict(), # 优化器的参数和buffer
    'loss': LOSS, # 损失函数的值
}, PATH)

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 设置学习率
checkpoint = torch.load(PATH)
Model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器的参数和buffer
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
#or
model.train()
#named_parameters()方法可以获取所有的参数，并且返回一个迭代器，可以遍历所有的参数
#使用parameter类可以直接将参数放入到模型中，而tensor类型不可以
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.register_parameter('mean',nn.Parameter(torch.zeros(1), requires_grad=True))
        self.pdf = torch.distributions.Normal(self.state_dict()['mean'],torch.tensor([1,0]))
    def forward(self, x):
        return -self.pdf.log_prob(x)
model = GaussianModel()
#requires_grad()方法可以设置参数是否需要反向传播(梯度更新)

#register_module()方法可以注册一个模块，
#get_submodule()方法可以获取一个子模块（就是一个层或者一个block）
#get_parameters()方法可以获取所有的参数,需要传入target参数以得到目标模块的路径，再根据路径来找到param_name

#buffer一般是一些附属的统计量
#_apply函数对子模块进行调用，对，parameter,buffer进行初始化,用于模型的初始化
#type函数可以转换数据类型
#to_empty()将模型移动到其他设备上，但是不会加载参数