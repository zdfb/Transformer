import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pre_process import Preprocess
from nets.transformer import transformer
from utils.utils_fit import fit_one_epoch

from nets.transformer_training import LabelSmoothing, NoamOpt


###### 开始训练 ######


class train_transformer():
    def __init__(self):
        super(train_transformer, self).__init__()

        self.data = Preprocess()  # 数据预处理

        self.d_model = 512
        self.loss_test_min = 1e9  # 初始化训练集loss

        self.src_vocab = len(self.data.en_word_dict)
        self.tgt_vocab = len(self.data.cn_word_dict)

        # 创建transformer模型
        self.model = transformer(self.src_vocab, self.tgt_vocab)

        self.criterion = LabelSmoothing(self.tgt_vocab, padding_idx = 0, smoothing=0)
        self.optimizer = NoamOpt(self.d_model, 1, 2000, torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
    
    def train(self, epoch):
        for i in range(epoch):
            test_loss = fit_one_epoch(self.data, self.model, self.criterion, self.optimizer)
            if test_loss < self.loss_test_min:
                self.loss_test_min = test_loss
                torch.save(self.model.state_dict(), 'transformer.pth')

if __name__ == "__main__":
    train = train_transformer()
    train.train(50)  # 训练50个epoch