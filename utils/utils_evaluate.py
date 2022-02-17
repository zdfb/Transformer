import torch
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from utils.utils import subsequent_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 定义正向推理流程 ######


# 写入结果
def write_result(data):
    file = open(f'result/result.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

# 产生结果
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # 定义开始标志
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])  # 产生输出概率
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim = 1)
    return ys

def evaluate(data, model):
    with torch.no_grad():
        for i in tqdm(range(len(data.test_en))):
            en_sent = " ".join([data.en_index_dict[w] for w in data.test_en[i]])
            write_result(en_sent)
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.test_cn[i]])
            write_result(cn_sent)

            src = torch.from_numpy(np.array(data.test_en[i])).long().to(device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len = 60, start_symbol = data.cn_word_dict["BOS"])  # 产生输出结果
            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()] # 取出词
                if sym != 'EOS':  # 若不为结束标志
                    translation.append(sym)
                else:
                    break
            write_result("translation: " + " ".join(translation) + "\n")