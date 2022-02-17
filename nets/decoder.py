import torch
import torch.nn as nn
from utils.utils import clones
import torch.nn.functional as F
from utils.utils_layers import SublayerConnection, LayerNorm


###### 定义Decoder部分 ######


# 定义每个Decoder层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # 输入维度
        self.self_attn = self_attn  # self_attention结构
        self.src_attn = src_attn  # cross_attention结构
        self.feed_forward = feed_forward  # 前向传播结构
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 第一层，self-attention结构
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # 第二层， cross-attention结构
        return self.sublayer[2](x, self.feed_forward)

# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)