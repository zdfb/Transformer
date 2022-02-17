import copy
import math
import torch
import torch.nn as nn
from utils.utils import clones
import torch.nn.functional as F


###### 定义self-attention机制 ######


# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // h  # 每个头的注意力机制中q、k的维度
        self.h = h  # 多头注意力机制的头数

        # 相当于四个输入为d_model, 输出为d_model的全连接层，未注册之间的forward关系
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def attention(self, query, key, value, mask = None, dropout = None):
        d_k = query.size(-1)  # q, k的维度长度

        # q.k^T / d_k ^ (1 / 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask
        p_attn = F.softmax(scores, dim = -1)  # 使用softmax将输入归一化至0～1范围内

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 增加一个维度

        n_batches = query.size(0)  # 每一个batch_size的样本数
        
        # 生成经过线性层转化后的q, k, v
        query, key, value = [linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for linear, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask = mask, dropout = self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)