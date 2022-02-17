import os
import torch
import numpy as np
import torch.nn as nn
from nltk import word_tokenize
import torch.nn.functional as F

from utils.pre_process import Preprocess
from nets.transformer import transformer
from utils.utils_evaluate import greedy_decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 定义翻译类 ######


class Translate():
    def __init__(self):
        super(Translate, self).__init__()

        self.model_path = 'data/transformer.pth'
        self.data = Preprocess()  # 数据预处理

        self.d_model = 512

        self.src_vocab = len(self.data.en_word_dict)
        self.tgt_vocab = len(self.data.cn_word_dict)

        # 创建transformer模型
        self.model = transformer(self.src_vocab, self.tgt_vocab)
        self.model.load_state_dict(torch.load(self.model_path, map_location = device))
        self.model = self.model.eval().to(device)

    
    # 读取数据
    def load_data(self, sentence):
        en = []
        en.append(["BOS"] + word_tokenize(sentence.lower()) + ["EOS"])  # 将英文进行切分， 增加开始与结尾标志
        return en
    
    # 将数据转化为id
    def wordToID(self, en, en_dict):
        out_en_ids = [[en_dict.get(w, 0) for w in en[0]]]
        return out_en_ids

    def translate(self, sentence):
        en = self.load_data(sentence)
        ids = self.wordToID(en, self.data.en_word_dict)

        with torch.no_grad():
            src = torch.from_numpy(np.array(ids)).long().to(device)

            src_mask = (src != 0).unsqueeze(-2)
            out = greedy_decode(self.model, src, src_mask, max_len = 60, start_symbol = self.data.cn_word_dict["BOS"])  # 产生输出结果
            translation = []
            for j in range(1, out.size(1)):
                sym = self.data.cn_index_dict[out[0, j].item()] # 取出词
                if sym != 'EOS':  # 若不为结束标志
                    translation.append(sym)
                else:
                    break
            print(" ".join(translation))