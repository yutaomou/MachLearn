import torch
import os
import numpy as np
from mpmath import eps
from torch import nn, optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.io import read_image
import math
torch.set_default_dtype(torch.float64)  # 设置默认形状为64位，避免类型转换

#mask可以避免padding中的值影响概率预测
#encoder中的mask:自身对自身的关联性计算
#decoder中的第一个mask:Masked Multi-head attention 也是序列本身对本身的关联性计算，mask保证因果性
#encoder memory（作为Key和value） 和decoder中的第一个mask Multi-head attention输出（作为Quary），它的mask涉及到两个不同的序列

#关于word embedding,以序列建模为例
#考虑source sentence 和 target sentence（离散建模）
#构建序列，序列的字符以在词表的索引显示

batch_size = 2

##定义单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8

##定义最大序列长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5 #所有样本的最大值为5

#src_len = torch.randint(2, 5, (batch_size, ))#元组中只包含一个元素时，需要在元素后面添加逗号
#tgt_len = torch.randint(2, 5, (batch_size, ))
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32) #定义具有两个元素的张量

##单词索引构成的句子，构建patch,设置pad
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len-L)), 0) \
                     for L in src_len]) #保证句子长度相同，列表形式，然后加维度，最后进行沿第0维拼接
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_tgt_seq_len-L)), 0) \
                     for L in tgt_len])

##构造embedding
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim) #+1是为了给padding编码，预留出索引为0的位置，第0行留给pad
tgt_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
src_embedding = src_embedding_table(src_seq) #每一个输入句子的编码
tgt_embedding = tgt_embedding_table(tgt_seq) #对实例后面直接加括号即调用forward方法

##构建position embedding(位置编码)
#构建时即可以用循环一个个填充，也可以直接矩阵相乘
#分为奇数列和偶数列，偶数列的编码为sin，奇数列的编码为cos
#formula：sin/cos(pos * 10000 ^ (2i / d_model))
#行矩阵pos
pos_mat = torch.arange(max_position_len).reshape((-1, 1)) #构建pos序列,并增加一个维度，将一维数据转化为二维数据
#列矩阵i
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1))/model_dim) #以10000为底，以2i/model_dim为指数
pe_embedding_table = torch.zeros(max_position_len, model_dim) # 初始化position embedding表
pe_embedding_table[:, 0::2] = torch.sin(pos_mat/i_mat) # 偶数列的编码为sin 利用了broadcast（广播机制）来使矩阵相乘后的维度达到一致
pe_embedding_table[:, 1::2] = torch.cos(pos_mat/i_mat) # 奇数列的编码为cos

pe_embedding = nn.Embedding(max_position_len, model_dim) #调用Embedding方法
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False) #权重中传入embedding_table

src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32) #_ 是占位符，代表一个随机变量
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos) #embedding的输入需要是pos——index而不是word——index
tgt_pe_embedding = pe_embedding(tgt_pos)


##构建Scaled dot product attention （scaled其实就是QK除一个数）
#注意力机制就是对于一个序列算出一个新的表征，表征是根据一个权重对于这个序列加权求和得到的
#nn.Linear(2, 3) #构建一个全连接网络，把2维映射到3维，也是一个表征，映射的权重是随机生成的
#根据Query和Kay的相似度算出权重，即对两个向量算内积（标量），再除(dk)^(1/2)（为了使softmax出来的数据方差减小），再softmax一下，即输出0-1间的数
#因为QK的点积方差为dk，所以除以标准差可以使分布更稳定

#encoder的self-attention mask
#mask的shape: [batch_size, max_src_len, max_src_len]（即QK的点积形状）, 值为1或负无穷(即需要关注的部分和不需要关注的部分（padding部分）)
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len]), 2) #得到一个有效的编码器的位置矩阵
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2)) #bmm: 三维矩阵相乘
print(valid_encoder_pos_matrix.shape) #得到batchsize * max(src_len) * max(src_len)的邻接矩阵，表示各个单词之间的关联性
#得到无效矩阵
invalid_encoder_pop_matrix = 1 - valid_encoder_pos_matrix #得到无效矩阵
#生成编码器的自注意力的mask矩阵，是一个布尔矩阵
mask_encoder_self_attention = invalid_encoder_pop_matrix.to(torch.bool) #得到mask矩阵，true的位置表示需要进行mask，false的位置表示不需要进行mask操作

#Q: (batch_size, seq_len, d_k)
#K: (batch_size, seq_len, d_k)
#Q 和 K 点积后的矩阵形状为 (batch_size, max(seq_len), max(seq_len))
score = torch.randn(batch_size, max(src_len), max(src_len)) #假设这个向量为Q*K的结果
masked_score = score.masked_fill(mask_encoder_self_attention, -1e9) #如果mask为True，填充为负无穷
prob = F.softmax(masked_score, -1) #再对mask算softmax
print(prob)



#softmax演示（scaled的重要性）
# alpha1 = 0.1
# alpha2 = 10
# score = torch.randn(5) #假设这个向量为Q*K的结果、
# prob1 = F.softmax(score*alpha1, -1) # *0.1会使softmax的结果差别变小
# prob2 = F.softmax(score*alpha2, -1) #如果×10，会使softmax的结果差别变大
# print(score)
# print(prob1)
# print(prob2)

#对于Jacobian矩阵
# def softmaxfunc(score):
#     return F.softmax(score)
# jacobian_mat1 = torch.autograd.functional.jacobian(softmaxfunc, score*alpha1)
# jacobian_mat2 = torch.autograd.functional.jacobian(softmaxfunc, score*alpha2) #对比*不同数值后的雅可比矩阵
# print(jacobian_mat1)
# print(jacobian_mat2)


#print(src_seq)
#print(tgt_seq)#batch_size = 2，每一行可以理解为一个句子
#print(src_embedding_table.weight)
#print(src_seq)
#print(src_embedding) #根据src_seq中的索引，从embedding表中取出对应索引的向量，第0行为padding向量
# print(pos_mat)
# print(i_mat)
# print(pe_embedding.weight)
# print(pe_embedding_table)



