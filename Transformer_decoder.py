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

##构造intra-attention的mask
#encoder的输出（memory）作为K，V输入到decoder中
#Q由目标序列生成
#公式：Q @ k^T shape: [batch_size, target_seq_len, src_seq_len] 和mask形状一样
#仿照encoder中的mask
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len]), 2) #得到一个有效的编码器的位置矩阵
print(valid_encoder_pos.shape)

valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len) - L)), 0) for L in tgt_len]), 2)
print(valid_decoder_pos.shape)

#目标序列对原序列的有效性的关系
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2)) #bmm表示含有batch_size的矩阵相乘,转置1，2维度表示不让batch_size的维度（第0维）参与计算
#得到[2,4,4]形状
print(valid_cross_pos_matrix)
#无效矩阵
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
#构造mask
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)
print(mask_cross_attention, "\n")

##构造decoder self attention mask
#decoder是以因果的方式进行填充
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L, L))), (0, max(tgt_len)-L, 0, max(tgt_len)-L)), 0) for L in tgt_len]) #先扩维，再对行，列都进行pad
print(valid_decoder_tri_matrix) #1st目标矩阵长度为4，生成4*4下三角矩阵，2nd目标矩阵长度为3，生成3*3下三角矩阵
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
print(invalid_decoder_tri_matrix)
score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix, -1e9) #将True的位置填充为负无穷
prob = F.softmax(masked_score, -1) #对最后一维进行softmax
print(prob) #生成注意力权重的矩阵 decoder_self_attention_mask

##构建self_attention
def scaled_dot_product_attention(Q, K, V, attetion_mask):
    #shape of Q, K, V : {batch_size*num_head, seq_len, model_dim/num_head}
    torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attetion_mask, -1e9) #将无效位置填充为负无穷
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context



