import copy
import torch
import numpy as np
import torch.nn as nn


###### 定义用到的工具函数 ######


# 复制某一层多次，加入到moduleList内
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 将输入数据转化为同个长度(序列最大长度), 不够补0
def seq_padding(X, padding = 0):
    L = [len(x) for x in X]  # 获取每一条数据的长度
    ML = max(L)  # 数据的最大长度
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

# 定义输入时的mask
def subsequent_mask(size):
    attn_shape = (1, size, size)  # self_attention 产生的score形状
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0