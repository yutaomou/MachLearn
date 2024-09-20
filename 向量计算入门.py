import torch
import numpy as np

torch.set_default_dtype(torch.float64)#设置默认形状为64位，避免类型转换

shape = (2,3,)#形状为（2，3）
x_data2 = torch.rand(shape)
print(x_data2)
x_data3 = torch.zeros(shape)
print(x_data3)
x_data4 = torch.ones(shape)
print(x_data4)

print(torch.chunk(x_data2,2))#分割成两个张量
c, d = torch.chunk(x_data2,2)

c, d = torch.chunk(x_data2,chunks = 2,dim=1)#按照列分割
print(c)
print(d)

torch.gather(x_data2,1,torch.tensor([[0,1,0],[1,0,1]]))#取值，逐行取对应列的值，dim=1代表沿着行进行

a = torch.arange(4.)
torch.reshape(a, (2, 2))#不改变元素顺序,变成二维矩阵

b = torch.tensor([0,1])
torch.reshape(b,(-1,))#不改变元素顺序,变成一维向量

#scatter_
tensor = torch.tensor([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6]])#原始张量
# 索引张量
index = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
# 源张量
src = torch.tensor([[-1, -2, -3, -4, -5], [10, 20, 30, 40, 50]])
# 使用 scatter_ 方法
tensor.scatter_(0, index, src)
print(tensor)#将src中对应位置的值赋值给tensor，实现对张量值的修改

#scatter_add_
#根据index索引，将src中对应位置的值加到tensor中对应位置的值上

#split
torch.split(x_data2,2)
#与chunk相比，split是均分的，而chunk是按比例划分的

#squeeze
#压缩张量,删除维度为1的维度
x_data5 = torch.tensor([[[1,2,3]],[[4,5,6]],[[7,8,9]]])
print(x_data5.shape)
x_data5.squeeze()
print(x_data5.squeeze().shape)

#stack
#沿着新维度将输入张量堆叠在一起
x_data6 = torch.rand([3,2])
x_data7 = torch.rand([3,2])
print(x_data6)
print(x_data7)
a = torch.stack([x_data6,x_data7],dim=0)#默认沿着0维堆叠
print(a)
print(a.shape)#输出为(2,3,2)

