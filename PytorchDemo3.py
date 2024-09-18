import torch
import numpy as np

torch.set_default_dtype(torch.float64)#设置默认形状为64位，避免类型转换

#transpose/swapaxes/swapdims
#用于交换张量中两个指定维度的位置
x_data1 = torch.rand([2,3,4])#形状为2*3*4的随机数张量
x_data1 = torch.transpose(x_data1,0,1)
print(x_data1.shape)#输出为3*2*4

#take
x_data2 = torch.tensor([[1,2,3],[4,5,6]])
index = torch.tensor([0,2,5])#将索引为0，2，5的元素取出（x_data2平铺）
print(torch.take(x_data2,index))

#tile
#将张量沿指定维度重复n次
x_data3 = torch.tensor([[1,2,3]])
print(x_data3.tile(2))
x_data4 = torch.tensor([[1,2],[3,4]])
print(torch.tile(x_data4,(2,2)))#对第一个维度（行）复制两份，对第二个维度(列)也复制两份

#unbind
#将张量沿指定维度拆分成多个张量
x_data5 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.unbind(x_data5,dim=0))#默认拆分维度为0

#unsqueeze
#在指定维度上增加一个维度
x_data6 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.unsqueeze(x_data6,dim=0))#形状由3*3变为1*3*3

#where
#条件满足时返回第一个参数，否则返回第二个参数
x_data7 = torch.randn(3,2)#生成-1 ~ 1 之间的随机数
x_data8 = torch.rand(3,2)#生成0 ~ 1 之间的随机数
a = torch.where(x_data7 > 0, x_data7, x_data8)#对于x_data7中大于0的元素，返回x_data7，否则返回x_data8
print(x_data7)
print(x_data8)
print(a)

#manual_seed
#设置随机数种子,使得在不同设备上进行随机得到的结果一样


#函数

#bernoulli
#以概率p生成0或1
x_data9 = torch.empty(3,3).uniform_(0,1)#生成0 ~ 1 之间的随机数
print(x_data9)
print(torch.bernoulli(x_data9))#根据概率生成0或1

#normal
#生成正态分布的随机数
x_data10 = torch.normal(mean=0,std=1,size=(3,3))



