#如何训练模型
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
from torchvision.io import read_image
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#训练模型需要不断迭代

#优化器有SGD等

#自定义数据集
class Custom_Image_Dataset(Dataset):  # 自定义数据集
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # 读取csv文件
        self.img_dir = img_dir  # 图片路径
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # 根据索引获取图片文件的路径
        image = read_image(img_path)  # 读取图片
        label = self.img_labels.iloc[idx, 1]  # 获取标签
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#定义dataloader
dataset1 = Custom_Image_Dataset(annotations_file='./data/train.csv', img_dir='./data/')#实例化数据集
train_dataloader = DataLoader(dataset1, batch_size=64, shuffle=True)

#编写卷积网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() #对父类进行调用
        self.flatten = nn.Flatten() #展平数据
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    def forward(self, x): #前向计算
        x = self.flatten(x) #展平数据
        logits = self.linear_relu_stack(x) # logits作为softmax的输入，得到一个概率值
        return logits

model = NeuralNetwork()

#定义超参数
learning_rate = 1e-3
batch_size = 64
epochs = 5

#设置目标函数
loss_fn = nn.CrossEntropyLoss() #分类函数用cross entropy

#构建优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #使用SGD算法
#每次训练前调用
optimizer.zero_grad()
#有梯度之后调用
optimizer.step()#更新参数

#训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #前向运算
        pred = model(X)
        loss = loss_fn(pred, y)

        #后向运算
        optimizer.zero_grad()
        loss.backward()#计算梯度
        optimizer.step()#更新参数

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") #每隔100次打印一次日志

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 计算正确率

        test_loss /= num_batches # 平均损失
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#开始训练
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(train_dataloader, model, loss_fn)
print("Done!")
