#autograd自动微分
from collections import OrderedDict

import torch
import os
import numpy as np
from torch import nn, optim
import torch.nn as nn
import pandas as pd
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#定义一个一层神经网络
x = torch.ones(5)#形状为1*5的一维全1矩阵
y = torch.zeros(3) #expected output
w = torch.randn(5,3, requires_grad=True) #随机初始化权重,设置为需要计算梯度
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b #矩阵乘法 = x * w +b #z是模型的实际输出
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) #计算z, y的交叉熵

#通过不断改变参数的值来降低loss,使用后向传播

#对根节点（在反向传播里就是最后的loss值）对父节点进行梯度运算
loss.backward() #当loss是一个标量时不需要传入参数，当loss是一个向量时，需要传入和loss形状相同的向量，如loss.backward(torch.ones(3))
print(w.grad)#梯度的shape和自身的shape一致
print(b.grad)
#backward一般只能调用一次，如果想调用多次的话需要loss.backward(retain_graph = True)

print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z_det = z.detach() #detach可以实现分离，保留梯度
print(z_det.requires_grad)
#由于梯度计算是累加的，所以需要调用zero_grad()来清楚梯度
w.grad.zero_()

inp = torch.eye(5, requires_grad=True) #定义一个5 * 5 的对角向量
out = (inp+1).pow(2) #先加一再平方
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(out),retain_graph=True)
print(inp.grad)

#在每次迭代无需清空梯度，调用optimizer.zero_grad()即可
#在求向量的梯度时，需要使用Jacobian矩阵
#Jacobian如何计算
def exp_reducer(x):#x是一个张量
    return x.exp().sum(dim=1)# 先对x进行exp（指数操作）操作，然后都x的第一维（按照行）求和
inputs = torch.randn(2, 2)#输入一个2*2的矩阵
y = jacobian(exp_reducer, inputs)#计算exp_reducer对inputs的雅克比矩阵，表示多元函数在某一点的局部线性化
print(y)
x = torch.randn(3)
def func(x):
    return x+x
print(jacobian(func, x))
#向量对向量的微分
a = torch.randn(3)
x = torch.randn(3,requires_grad=True)
y = func(x)
print(y.backward(torch.ones_like(y)))#计算各处的梯度
print(x.grad)#计算x的梯度
z = torch.ones_like(y)@jacobian(func, x)#计算x的梯度,即使用单位矩阵*对应的jacobian矩阵，@表示矩阵乘法

#矩阵对矩阵的微分
a = torch.randn(2,3,requires_grad=True)
b = torch.randn(3,2,requires_grad=True)
print(a)
print(a @ b)
y = a @ b
print(y.backward(torch.ones_like(y)))

def func1(a):
    return a @ b

print(jacobian(func1, a))
print(torch.ones_like(func1(a)) @ jacobian(func1,a))# 计算a的梯度

def func2(b):
    return a @ b
print(torch.ones_like(func2(b)) @ jacobian(func2,b)) # 计算b的梯度，和使用backward方法结果相同

print(torch.ones_like(b[:, 0]) @ jacobian(func, b[:, 0]))#计算b中第一列的梯度