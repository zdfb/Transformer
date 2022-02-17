import torch
import torch.nn as nn
import torch.nn.functional as F


###### 定义常用的层间操作 ######


# 定义layernorm
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # 放大参数
        self.b_2 = nn.Parameter(torch.zeros(features))  # 平移参数
        self.eps = eps  # 防止分母为0
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)  # 计算输入数据的平均值
        std = x.std(-1, keepdim = True)  # 计算输入数据的方差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 定义shortcut连接
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 先将输出经过layernorm
        # 再经过sublayer处理
        # 再通过dropout
        # 再通过shortcut连接
        return x + self.dropout(sublayer(self.norm(x)))

# 定义feedforward结构
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 产生最后输出的词
class Generator(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocabulary_size)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)