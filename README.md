# Transformer   Pytorch实现的简单英中机器翻译
## 部分机器翻译结果
``` bash
take care . 
照 顾 好 自 己 。

wait here . 
在 这 等 着 。

well done !
干 的 好 ！

he studied hard . 
他 努 力 學 習 。

he tends to lie . 
他 企 图 说 谎 。

he was very old . 
他 很 老 。

do you have kids ? 
你 們 有 孩 子 嗎 ？
```
## 预训练模型
+ .pth格式的预训练模型如下。<br>
>- 链接：https://pan.baidu.com/s/1hcuDdnFhFX0WceVNzCXBGA
>- 提取码：k1h0

## 训练自己的数据集
### 1. 按照data中train.txt所示格式准备数据集
英文与中文使用\t进行分割。
### 2. 开始训练
运行：
``` bash
python train.py
```
## 开始翻译
将训练好的模型，放置在data文件夹下。修改utils.utils_transformer中的model_path路径。
在translate.py文件下输入需要翻译的英文语句，运行：
``` bash
python translate.py
```

## 批量化测试
将训练好的模型，放置在data文件夹下。修改evaluate中的model_path路径。运行：
``` bash
python evaluate.py
```

## Reference
https://github.com/hinesboy/transformer-simple
