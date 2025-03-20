#kernal的数量 = in_channels * out_channels
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import floor

#调用pytorch中的卷积网络
in_channels = 1
out_channels = 1
kernel_size = 3 #如果为标量则代表是正方形，如果是元组，则代表为矩形
bias = False
batch_size = 1
input_size = [batch_size, in_channels, 4, 4] #4,4代表输入的图片大小
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")  #输入通道数，输出通道数，卷积核大小，填充,定义一个卷积层
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
# print(input_feature_map)
# print(output_feature_map)
# print(conv_layer.weight) #打印卷积层的kernel out_channels*in_channels*kernel_size

#同样，用functional中的conv实现
#需要手动制定weight和bias
output_feature_map1 = F.conv2d(input_feature_map, conv_layer.weight, bias=conv_layer.bias)

####################################################
#卷积核的滑动相乘实现二维卷积
input = torch.randn(5, 5)
kernal = torch.randn(3, 3)
#Step1 原始的矩阵运算实现二维卷积
def matrix_multiplication_conv2d(input, kernal, stride = 1, padding = 0, bias = 0):

    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding)) #更新input



    input_h, input_w = input.shape
    kernal_h, kernal_w = kernal.shape

    output_w = (floor((input_w - kernal_w) / stride) + 1) # 输出的宽度 floor 向下取整
    output_h = (floor((input_h - kernal_h) / stride) + 1) # 输出的高度
    output = torch.zeros(output_h, output_w) # 初始化输出矩阵
    for i in range(0, input_h - kernal_h + 1, stride): #对高度维进行遍历
        for j in range(0, input_w - kernal_w + 1, stride): #对宽度维进行遍历
            region = input[i:i+kernal_h, j:j+kernal_w] # 获取当前滑动窗口内的值,也是一个矩阵
            output[int(i/stride), int(j/stride)] = torch.sum(region * kernal) + bias#点乘并赋值

    return output

#矩阵运算的结果
mat_mul_conv_output = matrix_multiplication_conv2d(input, kernal)
print(mat_mul_conv_output)
#用Pytorch来验证是否正确
pytorch_api_conv_output = F.conv2d(input.reshape((1, 1, input.shape[0], input.shape[1])), \
         kernal.reshape((1, 1, kernal.shape[0], kernal.shape[1])))
print(pytorch_api_conv_output.squeeze(0).squeeze(0)) #转化为2维

#Step2:考虑batchsize和channel维度
def matrix_multiplication_for_conv2d_full(input, kernal, bias=0, stride=1, padding=0):
    #input,kernal都是4维的
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))
    bs, in_channel, input_h, input_w = input.shape
    out_channel, in_channel, kernal_h, kernal_w = kernal.shape
    if bias is None:
        bias = torch.zeros(out_channel)

    output_h = (math.floor((input_h-kernal_h)/stride) + 1) #输出的高度
    output_w = (math.floor((input_w-kernal_w)/stride) + 1) #输出的宽度
    output = torch.zeros(bs, out_channel, output_h, output_w) #初始化输出矩阵

    for ind in range(bs):
        for oc in range(out_channel):
            for ic in range(in_channel):
                for i in range(0, input_h - kernal_h + 1, stride):
                    for j in range(0, input_w - kernal_w + 1, stride):
                        region = input[ind, ic, i:i+kernal_h, j:j+kernal_w] # 获取当前滑动窗口内的值,是一个矩阵
                        output[ind, oc, int(i/stride), int(j/stride)] += torch.sum(region * kernal[oc, ic]) + bias[oc] #点乘并赋值

    return output





#转置卷积是根据普通卷积的反向传播算出来的
#把kernal转置一下，然后和output相乘，就可以恢复原始的input，从小图片到大图片
pytorch_trasnspose_conv_output = F.conv_transpose2d()
#转置卷积用于放大图像分辨率，从特征映射到空间
#在 CNN 中，kernel 的权重是通过反向传播算法自动学习的。
#初始时，kernel 的权重通常是随机初始化的，然后通过训练过程不断调整，使其能够提取出对任务（如分类或检测）最有用的特征



