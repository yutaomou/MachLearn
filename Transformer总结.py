"""
Transformer模型总结
Seq to Seq模型，之前RNN也是SEQ2SEQ 但是RNN递归计算，所以长时建模差，串行计算效率低 Transformer解决这两个缺点 （无先验假设的Self attention， 完全由DNN（Deep Neural Network）构成，无递归计算，可以并行计算）
                                                 |Position-embedding  DNN没有考虑位置信息，所以引入PE
                                                |
                                                Mult-head Self-attention （一般是8头自注意力）  核心，计算量最大 对任意两两字符计算关联性 （对于PE序列计算Multiheadattention的表征）
                                               |
                                |Encoder ---- |   LayerNorm & Residual   层归一化+残差连接（MultiHeadattention输出的序列进行层归一化+word-embedding）
                                |               |                                        | Linear1(Large)  （2048维）
                               |                 | Feedforward Neural Network（FNN）---- | （类比与CNN中的通道分离算法类似，Self-attention实现位置混合，FNN对特征层进行混合）
                              |                  |                                       | Linear2(d_model) （512维）
                              |                  | LayerNorm & Residual  第一个残差连接的结果与FNN的结果（层归一化之后）进行残差连接
                              |                    得到第一个encoder Block的输出 把n个block堆叠起来就可以了
                  |模型结构---- |
                |              |
                |               |                     Position Embedding 对目标序列
                                                    |
                |               |                   Casual Multi-head Self-attention 因果自注意力（masked） input = PE + WE  （目标序列的PE和WE） 训练阶段，输入为真实的tgt_embedding 推理阶段，输入为上一步预测出来的embedding。自回归模型的训练和推理有一定差别
                                                   |
                |               |                 | LayerNorm & Residual
                                                  |
                |               |                |  Memory-base Multi-head Cross-attention 上一次decoder的输出作为Q，encoder的输出作为Memory和Key，算出attention
                                                |
                |               |Decoder --- |      LayerNorm & Residual
                                                |                                |Linear1(Large)
                |                                  |FeedForward Neural Network--| #FNN做特征上的混合，MultiHead attention做位置上的混合，组合起来使计算量更小
                |                                  |                             |Linear2(d_model)
                |                                   | LayerNorm & Residual
                |
                |
                |
                |            | Encoder Only     ----BERT 分类任务  非流式任务（不需要每次返回一小部分给用户）
                |           |
Transformer----| --使用类型--|   Decoder Only     ----GPT系列  语言建模  自回归生成任务  流式任务
                |           |
                |            |  Encoder-Decoder  ----机器翻译  语音识别
                |
                |
                |
                |
                |
                |             |   无先验假设 （例如：局部关联性，有序性建模）与CNN差别较大，任意位置和其他位置都可以有关联性 好处：相比CNN，RNN，更快学到无论长时建模还是短时关联 缺点： 先验假设越多，对数据量要求越低。
                |            |
                |--特点 -----|     核心在于自注意力机制，平方复杂度  自注意力机制特点：训练越长的序列，复杂度与序列长度的平方成正比 如果要避免对所有位置进行计算（降低自注意力机制的计算量），则要引入先验假设
                             |
                              |   数据量的要求与先验假设的程度成反比 需要大量数据


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#机器翻译（分类任务）的Loss计算
logits = torch.randn(2, 3, 4) #batchsize = 2, seq_length = 3, vocab_size = 4 代表预测出的结果
label = torch.randint(0, 4, (2, 3)) #对每一个位置进行标签
logits = logits.transpose(1, 2) #转置logits为pytorch要求的形状
F.cross_entropy(logits, label) #计算平均交叉熵
F.cross_entropy(logits, label, reduction='none') #计算每个单词的交叉熵





