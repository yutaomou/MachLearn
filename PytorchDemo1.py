import torch
import numpy as np
from torch import eye

torch.set_default_dtype(torch.float64)#设置默认形状为64位，避免类型转换

data1 = [[1,2],[3,4]]
x_data1 = torch.tensor(data1)#创建张量
print(x_data1)
print(type(x_data1))
print(x_data1.shape)

shape = (2,3,)#形状为（2，3）
x_data2 = torch.rand(shape)
print(x_data2)
x_data3 = torch.zeros(shape)
print(x_data3)
x_data4 = torch.ones(shape)
print(x_data4)

#从原有的张量初始化，形状相同
x_data5 = torch.rand_like(x_data2)
print(x_data5)
x_data6 = torch.ones_like(x_data2)
print(x_data6)
x_data7 = torch.zeros_like(x_data2)
print(x_data7)

print(x_data1.shape)
print(x_data1.dtype)
print(x_data1.device)#一般默认运行在cpu上

print(torch.arange(5))#遍历0-4，默认从0开始，步长为1
print(torch.arange(0,5,2))

print(torch, eye(3))#创建单位矩阵

print(torch.cat([x_data2,x_data3]))#拼接张量,形状必须相同
print(torch.cat([x_data2,x_data3],dim=1))#拼接张量，指定拼接方向





