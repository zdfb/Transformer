import torch
import torch.nn as nn
from utils.utils import clones
import torch.nn.functional as F
from utils.utils_layers import SublayerConnection, LayerNorm


###### 定义Encoder部分 ######


# 定义每个Encoder层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attn  # self_attention部分
        self.feed_forward = feed_forward  # feed_forward部分
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.size = size  # 输入维度

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))  # self_attention部分
        return self.sublayer[1](x, self.feed_forward)  # feed_forward部分

# 定义Encoder部分
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) 