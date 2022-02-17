import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pre_process import Preprocess
from nets.transformer import transformer
from utils.utils_evaluate import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 测试transformer ######


class evaluate_transformer():
    def __init__(self):
        super(evaluate_transformer, self).__init__()

        self.model_path = 'data/transformer.pth'
        self.data = Preprocess()  # 数据预处理

        self.d_model = 512

        self.src_vocab = len(self.data.en_word_dict)
        self.tgt_vocab = len(self.data.cn_word_dict)

        # 创建transformer模型
        model = transformer(self.src_vocab, self.tgt_vocab)
        model.load_state_dict(torch.load(self.model_path, map_location = device))
        model = model.eval()

        self.model = model.to(device)
    
    def test(self):
        evaluate(self.data, self.model)

if __name__ == "__main__":
    test = evaluate_transformer()
    test.test()