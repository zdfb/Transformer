import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from utils.utils import seq_padding, subsequent_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### 定义数据预处理 ######


class Preprocess:
    def __init__(self):
        self.train_file = 'data/train.txt'
        self.test_file = 'data/test.txt'

        self.batch_size = 64

        self.UNK = 0
        self.PAD = 1

        # 读取数据并进行分词
        self.train_en, self.train_cn = self.load_data(self.train_file)
        self.test_en, self.test_cn = self.load_data(self.test_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # 转化为id
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.test_en, self.test_cn = self.wordToID(self.test_en, self.test_cn, self.en_word_dict, self.cn_word_dict)

        # 划分batch，增加padding处理与mask处理
        self.train_data = self.splitBatch(self.train_en, self.train_cn, self.batch_size)
        self.test_data = self.splitBatch(self.test_en, self.test_cn, self.batch_size)

    
    # 读取数据
    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r') as f:
            # 读取每行
            for line in f:
                line = line.strip().split('\t')  # 按照\t进行切分

                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])  # 将英文进行切分， 增加开始与结尾标志
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"]) # 将中文进行切分， 增加开始与结尾标志
            return en, cn
    
    # 建立词典
    def build_dict(self, sentences, max_words = 50000):
        word_count = Counter()  

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1  # 记录每个词出现的次数
        
        ls = word_count.most_common(max_words)  # 获取前max_words频次的词
        total_words = len(ls) + 2  # 增加两个特殊词UNK与PAD

        word_dict = {w[0]:index + 2 for index, w in enumerate(ls)}  # 建立词与index相对应的词表

        word_dict['UNK'] = self.UNK
        word_dict['PAD'] = self.PAD

        index_dict = {v:k for k, v in word_dict.items()}  # 转换为index: word形式
        return word_dict, total_words, index_dict
    
    # 将Word转化为ID
    def wordToID(self, en, cn, en_dict, cn_dict, sort = True):
        length = len(en)  # 中文与英文的输入句子数

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]
        
        # 将输入序列按照英文长度进行排序
        def len_argsort(seq):
            return sorted(range(len(seq)), key = lambda x: len(seq[x]))
        
        # 把中文和英文按照同样的顺序排列
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
        
        return out_en_ids, out_cn_ids
    
    # 划分batch，并增加padding与mask
    def splitBatch(self, en, cn, batch_size, shuffle = True):
        idx_list = np.arange(0, len(en), batch_size)  # 每个batch的起始id
        if shuffle:
            np.random.shuffle(idx_list) # 打乱排列
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]

            # 长度不够的位置补0
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
        return batches


# 对每个batch的数据进行预处理
class Batch:
    def __init__(self, src, trg = None, pad = 0):
        src = torch.from_numpy(src).to(device).long()
        trg = torch.from_numpy(trg).to(device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # 将padding部分进行遮挡
        if trg is not None:
            self.trg = trg[:, :-1]  # 删除结束标志
            self.trg_y = trg[:, 1:]  # 删除开始标志
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask        
