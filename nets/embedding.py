import math
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 定义embedding层与position_embedding机制 ######


# embedding层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# Position embedding机制
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model, device = device)
        position = torch.arange(0., max_len,  device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2,  device = device) *- (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 加1个维度
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # positon embedding 不参与梯度回传
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)  