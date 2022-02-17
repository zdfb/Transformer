import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.encoder import Encoder, EncoderLayer
from nets.decoder import Decoder, DecoderLayer
from nets.self_attention import MultiHeadAttention
from nets.embedding import PositionalEncoding, Embeddings
from utils.utils_layers import PositionwiseFeedForward, Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### 定义Transformer结构 ######


# 定义Transformer
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def transformer(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model).to(device)  # 定义self-attention结构
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)  # 定义feedforward层
    position = PositionalEncoding(d_model, dropout).to(device)  # 位置编码

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(device), N).to(device)  # encoder结构
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(device), N).to(device)  # decoder结构

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab).to(device), c(position))  # Encoder方向的embedding
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab).to(device), c(position))  # Decoder方向的embedding

    generator = Generator(d_model, tgt_vocab).to(device)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)    