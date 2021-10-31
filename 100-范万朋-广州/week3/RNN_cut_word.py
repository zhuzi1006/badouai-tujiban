# coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        """
        :param input_dim: 输入维度：50
        :param hidden_size: 隐藏层神经元：100
        :param num_rnn_layers: rnn层数：3
        :param vocab: 字符表
        """
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)  # shape=(vocab_size, dim)
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                num_layers=num_rnn_layers,
                                nonlinearity="relu",
                                dropout=0.1)
        self.classify = nn.Linear(hidden_size, 2)
        # 标签为-100的不计入Cross的计算，只计算标签为0或1的
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):  # x,y = [20, 20]
        x = self.embedding(x)  # x = [20, 20, 50] => [20个句子，20个字，每个字50个维度]
        x, _ = self.rnn_layer(x)  # x = [20, 20, 100]
        y_pred = self.classify(x)  # y_pred = [20, 20, 2] => [20个句子，20个字，每个字是不是词的概率]
        if y is not None:
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred


class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.data = []
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length  # 20
        self.load()

    def load(self):
        """
        数据加载，最终data的维度是[10000, 20, 20]
        第一维: 数据量
        第二维：字符串的数字序列
        第三维：标签序列
        """
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)  # 文本转为数字序列,19
                label = sequence_to_label(line)  # 生成标签,19
                sequence, label = self.padding(sequence, label)  # 20，保证长度都为20
                sequence = torch.LongTensor(sequence)  # torch.Size([20])
                label = torch.LongTensor(label)  # torch.Size([20])
                self.data.append([sequence, label])
                if len(self.data) >= 10000:
                    # 每个样本数据不超过1w+
                    break

    def padding(self, sequence, label):
        """
        数据补齐，保证每段数据长度都是20，不足的用0填充
                标签长度也是20，不足的用-100填充
        :param sequence: 字符串的数字序列，长度不会大于20
        :param label: 字符串对应的标签序列
        """
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))  # 在长度不足20的字符串末尾补0
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))  # 在长度不足20的标签末尾补-100
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    # get(key, default=None) 返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence


# 基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)  # 词的前缀、非词的设置为0
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1  # 词的末尾设置为1
    return label


# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab


# 建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    """
    :param corpus_path: 语料集
    :param vocab: 字符字典 {‘啊’：1，‘额’：2，...}
    :param max_length: 样本最大长度，20
    :param batch_size: 每次训练样本个数，20
    """
    dataset = Dataset(corpus_path=corpus_path, vocab=vocab, max_length=max_length)  # diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)  # torch
    return data_loader


def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    max_length = 20  # 样本最大长度
    learning_rate = 1e-3  # 学习率
    vocab_path = "chars.txt"  # 字表文件路径
    corpus_path = "corpus.txt"  # 语料文件路径

    vocab = build_vocab(vocab_path=vocab_path)  # 建立字表

    data_loader = build_dataset(corpus_path=corpus_path, vocab=vocab, max_length=max_length,
                                batch_size=batch_size)  # 建立数据集

    model = TorchModel(input_dim=char_dim, hidden_size=hidden_size,
                       num_rnn_layers=num_rnn_layers, vocab=vocab)  # 建立模型

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    # 训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:  # x,y = torch.Size([20, 20]),20个句子，对应20组标签，20个字，每组20个标签
            optim.zero_grad()  # 梯度归零
            loss = model.forward(x=x, y=y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 最终预测
def predict(model_path, vocab_path, input_strings):
    # 配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    word = []  # 切分好的字符串

    vocab = build_vocab(vocab_path)  # 建立字表

    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)  # 建立模型

    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重

    model.eval()  # 测试模式，不使用Dropout
    for input_string in input_strings:
        temp_word = []
        # 逐条预测
        x = sentence_to_sequence(input_string, vocab)  # 字符串的数字数列
        with torch.no_grad():  # 不计算梯度
            result = model.forward(torch.LongTensor([x]))[0]  # [7, 2]，【7个字，每个字是否为词的末尾】
            result = torch.argmax(result, dim=-1)  # 预测出的01序列，[7]
            # 在预测为1的地方切分，将切分后文本打印出来
            sign = 0  # 标记，记录上一次切分的位置
            for index, p in enumerate(result):
                if p == 1:
                    temp_word.append(input_string[sign:index + 1])
                    sign = index + 1  # 更新切分位置
                if len(result) == index + 1 and p == 0:
                    temp_word.append(input_string[sign:])
            word.append(temp_word)
    return word

#
# if __name__ == "__main__":
#     # print(jieba.lcut("今天天气不错我们去春游吧"))
#     # print(sequence_to_label("今天天气不错我们去春游吧"))
#     # main()
#     # input_strings = ["同时国内有望出台新汽车刺激方案",
#     #                  "沪胶后市有望延续强势",
#     #                  "经过两个交易日的强势调整后",
#     #                  "昨日上海天然橡胶期货价格再度大幅上扬，成交价达到二十五万"]
#     input_strings = ["经常有意见分歧"]
#     predict("model.pth", "chars.txt", input_strings)
