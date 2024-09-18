#parameters,modules,state_dict源码刨析
import torch
import os
import numpy as np
from torch import nn, optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#save model
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weight.pth')#保存buffer,parameter等变量
#更标准的保存方法
EPOCH = 5
PATH = './model.pth'
LOSS = 0.4
optimizer = optim.SGD(model.parameters(), lr=0.001)

torch.save({
    'epoch': EPOCH,#保存训练周期
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),#保存优化器的量
    'loss': LOSS,
})

#load model
model = models.vgg16()#创建一个空的模型(类型和要加载的模型一致)
model.load_state_dict(torch.load('model_weight.pth'))#state_dict是module中的一个成员，叫状态字典，存储了模型的参数
model.train()
#更标准的加载方法
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()

#to函数，将数据移动到其他设备
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
        self.batch_norm = torch.nn.BatchNorm2d(4)#batch_norm层标准化每一批次数据的分布来加速训练过程并提高模型稳定性,减少过拟合

test_module = MyModel()
print(test_module.__modules)
test_module.to(torch.double)#将模型中的数据类型改为double类型
test_module._modules['linear1'].weight.dtype#查看修改后的数据类型
test_module.parameter()#查看parameter参数,需要使用_register_parameter后才能使用
test_module.state_dict()#可以查看所有的参数和buffer等

#_name_numbers可以查找module1的任意东西
for p in test_module.parameters():
    print(p) #可以查看每一个线性层的weight和bias
