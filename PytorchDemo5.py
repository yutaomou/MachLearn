# 在pytorch中搭建分类网络
import torch
import os
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

# transforms
# 对dataset中读取的标签和样本进行数据预处理,使符合模型规范

# Build the neural network
# torch.nn提供了构建神经网络所需的所有模块，每一个模块都是nn.Module的子类

# 检查是否使用gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using {device}'.format(device=device))


# 定义神经网络模块
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 展平数据(具体实现见下文)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 线性层  第一个参数代表输入的维度，第二个参数代表输出的维度
            nn.ReLU(),  # relu非线性函数（激活函数）
            nn.Linear(512, 512),  # 线性层
            nn.ReLU(),  # relu非线性函数（激活函数）
            nn.Linear(512, 10),  # 线性层 （10个类型的分类）
        )  # 构建网络，按照顺序传入层,自动串联起来

    def forward(self, x):
        x = self.flatten(x)  # 展平数据
        logits = self.linear_relu_stack(x)
        return logits


# 实例化模块
model = NeuralNetwork().to(device)
print(model)  # 看model的每一个子模块以及输入输出大小

# 使用模块,调用forward方法
x = torch.rand(1, 28, 28, device=device)  # 创建随机张量
logits = model(x)  # 作为forward的参数传入,得到最后线性层的输出,logits（能量）作为softmax层的一个输入
pred_probab = nn.Softmax(dim=1)(logits)  # 实例化softmax层，对第一维进行归一化，得到一个预测的概率
y_pred = pred_probab.argmax(1)  # 得到概率的最大值的索引，即预测的类别
print(f"Predicted class: {y_pred}")

# model layers
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# Flatten层
input = torch.randn(32, 1, 5, 5)
m = nn.Sequential(
    nn.Conv2d(1, 32, 5, 1, 1),
    nn.Flatten()
)
output = m(input)
print(output.size())

flatten = nn.Flatten()  # 展平数据,只保留batch_size和feature_dim(所有其他维度相乘的大小)
flat_image = flatten(input_image)
print(flat_image.size())

# Linear层
layer1 = nn.Linear(in_features=28 * 28, out_features=20)  # 输入28*28维，输出20维
hidden1 = layer1(flat_image)  # flat_image的维度为28*28
print(hidden1.size())

# ReLU(非线性层,将矩阵中的负数变为0)
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequential
# 是一个ordered container
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)  # 使这些数据有序地经过这些模块
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# Softmax(归一化分类)
softmax = nn.Softmax(dim=1)  # 实例化
pred_probab1 = softmax(logits)

