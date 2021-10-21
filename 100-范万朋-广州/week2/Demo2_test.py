"""
第二次作业，对原来的代码进行了魔改
①把神经网络的层次单独封为一个类 DNNNet并修改
    1.将隐藏层的激活函数由 sigmoid 替换为 ReLu
    2.将输出层的激活函数由 sigmoid 替换为 softmax
②把神经网络的训练部分单独封为一个类 Model
③对 Model类 进行了健壮性的扩充
    1.可选择多种损失函数、多种优化器
    2.将【训练 train】代码跟【测试 evaluate】代码独立封装
④修改了词汇表，添加了中文字符【朋】
⑤修改了B类样本的条件
    1.从不含abc且含有xyz 改为 不含abc且含有【朋】
⑦预测结果的正确率保持在 99% 左右
"""
# -*- codeing = utf-8 -*-
# @Time: 2021/10/8 19:08
# @Author: 棒棒朋
# @File: Demo2_test.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import json

"""******************* 【一、数据的生成，包括测试、训练、以及标签数据】*******************"""


def build_vocab():
    """
    :return:词汇字典：{"a":1, "b":2, "c":3...}
    """
    chars = "abcdefghijklmnopqrstuvwxyz朋"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab


def build_sample(sentence_length):
    """
    随机生成一个样本，从所有字中选取sentence_length个字，并且生成标签集
    :param sentence_length: 单个样本的长度，比如6个字符
    :return: 样本集、标签集
    """
    vocab = build_vocab()  # 建立词汇表
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # A类样本
    if set("abc") & set(x) and not set("xyz") & set(x):
        y = 0
    # B类样本
    # elif not set("abc") & set(x) and set("xyz") & set(x):
    elif not set("abc") & set(x) and set("朋") & set(x):
        y = 1
    # C类样本
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, sentence_length):
    """
    输入需要的样本数量，建立数据集
    :param sample_length: 样本个数
    :param sentence_length: 单个样本的长度，比如6个字符
    :return: tensor格式的数据、tensor格式的标签
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


"""*******************          【二、神经网络的模型搭建】        *******************"""


class DNNNet(nn.Module):
    def __init__(self):
        super(DNNNet, self).__init__()
        self.linear1 = nn.Linear(20, 30)
        self.linear2 = nn.Linear(30, 3)
        self.dropout = nn.Dropout(0.2)
        self.max_pool = nn.MaxPool1d(6)
        self.embedding = nn.Embedding(29, 20)

    def forward(self, x):  # [20,6]
        x = self.embedding(x)  # [20,6,20]
        x = F.relu(self.linear1(x))  # [20,6,30]
        x = self.max_pool(x.transpose(1, 2)).squeeze()  # [20,30]
        x = self.dropout(x)
        x = F.softmax(self.linear2(x), dim=1)  # [30,3]
        return x


class Model:
    def __init__(self, net, loss_name, optimist_name, epochs):
        self.net = net
        self.loss = self.creat_loss_fun(loss_name)
        self.optimist = self.creat_optimist_fun(optimist_name)
        self.epochs = epochs

    def creat_loss_fun(self, loss_name):
        loss_fun_dirt = {'CROSS': nn.CrossEntropyLoss(), 'MSE': nn.MSELoss()}
        return loss_fun_dirt[loss_name]

    def creat_optimist_fun(self, optimist_name):
        optimist_fun_dirt = {'SGD': torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9),
                             'ADAM': torch.optim.Adam(self.net.parameters(), lr=0.01, betas=(0.9, 0.999))}
        return optimist_fun_dirt[optimist_name]

    def train(self):
        for epoch in range(self.epochs):
            for i in range(100):
                data, labels = build_dataset(sample_length=20, sentence_length=6)
                self.optimist.zero_grad()
                output = self.net.forward(data)
                loss = self.loss(output, labels.squeeze())
                loss.backward()
                print("第%d轮，loss=" % i, loss)
                self.optimist.step()

    def evaluate(self):
        total = 0
        error = 0
        correct = 0
        # vocab = build_vocab()  # 构建词汇表
        with torch.no_grad():  # 测试和预测的时候不需要使用梯度
            text, labels = build_dataset(sample_length=200, sentence_length=6)  # 建立200个用于测试的样本
            output = self.net.forward(text)
            pred = torch.argmax(output, dim=1, keepdim=True)
            total += len(labels)
            error += (pred != labels).sum().item()
            correct += (pred == labels).sum().item()
        print('测试的正确率: %d %%' % (100 * correct / total))
        print("总共 %d 张. ;正确预测了：%d ;错误预测了：%d " % (total, correct, error))


"""*******************          【三、最终预测】        *******************"""


def predict(model_path, input_strings):
    """
    :param model_path: 模型路径
    :param input_strings: 预测数据
    """
    vocab = build_vocab()  # 构建词汇表
    net = DNNNet()
    net.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        temp = []
        for char in input_string:
            try:
                temp.append(vocab[char])  # 将输入序列化
            except:
                temp.append(28)
        x.append(temp)
    with torch.no_grad():  # 不计算梯度
        result = net.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print(int(torch.argmax(result[i])), input_string, result[i])  # 打印结果


if __name__ == '__main__':
    net = DNNNet()
    model = Model(net, "CROSS", "ADAM", 5)
    model.train()  # 训练
    model.evaluate()  # 测试
    torch.save(net.state_dict(), "Model_test.pth")  # 保存模型

    test_strings = ["juvxee", "arwarg", "rbweqg", "n朋ddww", "abcxyz"]
    predict(model_path="Model_test.pth", input_strings=test_strings)
