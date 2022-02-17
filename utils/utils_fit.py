import time
import torch
from nets.transformer_training import SimpleLossCompute


###### 定义训练一个epoch ######


def fit(data, model, loss_compute):
    total_tokens = 0
    total_loss = 0

    for step, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        # 画进度条
        rate = (step + 1) / len(data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, total_loss / total_tokens), end="")
    print()
    return total_loss / total_tokens

def fit_one_epoch(data, model, criterion, optimizer):
    start_time = time.time()  # 获取当前时间
    model.train()
    fit(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer))

    model.eval()
    loss = fit(data.test_data, model, SimpleLossCompute(model.generator, criterion, None))
    print('total_test_loss: %3f, epoch_time : %3f.'%(loss, time.time() - start_time))
    return loss  