# coding = utf-8
# author:will

import  torch
import torch.nn as nn
import json
import random
import numpy as np
import matplotlib.pyplot as plt


"""
基于pytorch实现一个网络完成一个简单任务，判断文本中是否有某些特定字符
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.layer(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pool(x.transpose(1,2).squeeze())
        x = self.classify()
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"   #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1    #每个字符对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

vocab = build_vocab()
# print(vocab)

#随机生成一个样本，从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可以重复
    x = [random.choice(list(vocab.keys()))]     #random.choice() 返回一个列表，元组或字符串的随机项。

    #指定哪些字出现时是正样本
    if set("abc") & set(x):
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #把字转换成序号，为了做embedding
    return x, y

# a = build_sample(vocab, 6)
# print(a)


#建立数据集，输入需要的样本量。
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码，用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本" %(sum(y), 200 - sum(y)))
    correct, worng = 0, 0
    with torch.no_grad():
        y_pred = model(x)     #模型预测
        for y_p, y_t in zip(y_pred, y):   #与真是标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1     #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1    #正样本判断正确
            else:
                worng += 1
    print("正确预测个数：%d，正确率：%f" %(correct, correct/(correct+worng)))
    return correct/(correct+worng)


if __name__ == "__main__":
    main()