# Dataset&DataLoader
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

# dataset处理单个数据，从磁盘读取它的特征和标签，进行函数的预处理，转变为x，y的训练对
training_data = datasets.FashionMNIST(
    root="D:\桌面",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="D:\桌面",
    train=False,
    download=True,
    transform=ToTensor()
)


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

# dataloader处理多个数据，将数据集分成多个batch，每次从batch中提取数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=False,collate_fn=None)#shufflle用于打乱数据集,是官方默认的一种sampler。当不需要梯度更新时，shuffle=False
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

#sampler和shuffer互斥，不能同时设置
# batch_sampler设置之后不能再设置batchsize，shuffer，sampler，droplast

#collate_fn：将多个数据拼接成一个batch
# collate_fn=None 默认的collate_fn (什么都不干)
# __iter__ 实例化时，返回一个迭代器，迭代器返回的是一个batch



"""
sum
1.自定义一个dataset，加载特征文件和标签文件，可以对单个特征/标签及进行预处理
2.把dataset放入dataloader中，dataloader负责把单个样本拼成minibatch。sampler决定取样本的顺序，if shuffe = True并且不设置sampler，样本顺序随机，否则顺序不变（单样本级别）
batch_sampler决定batch的顺序，if shuffle = True并且不设置batch_sampler，batch顺序随机，否则顺序不变（batch级别）
collate_fn对batch进行填充，保持统一长度（进行后处理）
调用dataloader时一般使用__iter__方法，next（iter（train_dataloader）），基于mini_batch的index返回一批数据(枚举)
"""

