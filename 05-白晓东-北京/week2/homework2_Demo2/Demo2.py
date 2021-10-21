#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.output_dim = 40
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)  # 输入字表长度， 输入特征维度
        self.layer = nn.Linear(input_dim, self.output_dim)                     # 输入维度和输出维度
        self.pool_0 = nn.MaxPool1d(2)
        self.rnn = nn.RNN(input_size=40, hidden_size=40, batch_first=True)  # 输入和输出的特征维度
        self.pool = nn.MaxPool1d(3)  # 一维的pool窗口大小
        self.classify = nn.Linear(40, 4)
        self.activation = torch.sigmoid   # sigmoid做激活函数最终的正确率不超过0.6，使用relu激活函数也很差，难道是采用rnn的原因？
        self.dropout = nn.Dropout(0.7)
        self.loss = nn.functional.cross_entropy  # loss采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)   # 输入 20*6， 输出20*6*20
        x = self.layer(x)       # 输出 20*6*40
        x = self.pool_0(x.transpose(1, 2))  # 输出 20*40*3
        x = self.dropout(x)
        x = self.activation(x)  #
        x, h = self.rnn(x.transpose(1, 2))  # 输出 20*3*40
        x = self.pool(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)，输出20*40
        x = self.classify(x)    # input shape:(batch_size, input_dim)  输出 20*3
        y_pred = self.activation(x)               # input shape:(batch_size, 3) (20,3)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


def build_vocab():
    """
    字符集随便挑了一些汉字，实际上还可以扩充
    为每个字生成一个标号
    {  "a":1, "b":2, "c":3...}
    abc -> [1,2,3]
    """
    chars = "aabcdefghijklmnopqrstuvwxyz我爱编程编程使我快乐在badou提升自己0123456789"  # 字符集
    vocab = {}
    index = 1
    for char in chars:
        if vocab.get(char, 0) == 0:
            vocab[char] = index  # 每个字对应一个序号
            index += 1
    vocab['unk'] = len(vocab) + 1
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # A类样本
    chinese_num = 0
    for ch in x:
        # 当存在三个汉字时，就是A类样本
        if ch != 'unk' and ord(ch) > 255:
            chinese_num += 1
            if chinese_num >= 3:
                y = 0
                break
    else:
        # B类样本
        if set("abc") & set(x) and not set("xyz") & set(x):
            y = 1
        # C类样本
        elif not set("abc") & set(x) and set("xyz") & set(x):
            y = 2
        # D类样本
        else:
            y = 3
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200  # 测试样本数量
    x, y = build_dataset(total, vocab, sample_length)   # 建立200个用于测试的样本
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d" % (y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1   # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, total, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    epoch_num = 15        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 1000   # 每轮训练总共训练的样本总数
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    vocab = build_vocab()       # 建立字表
    model = build_model(vocab, char_dim, sentence_length)    # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)   # 建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构建一组训练样本 输入 20*6 （batchsize, seqlength）
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # plt.plot(range(len(log)), [l[0] for l in log])  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log])  # 画loss曲线
    # plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 最终预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)    # 建立模型
    model.load_state_dict(torch.load(model_path))       # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式，不使用dropout
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print(int(torch.argmax(result[i])), input_string, result[i]) # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["juvxee", "yrwfrg", "rbweqg", "nlhdww"]
    predict("model.pth", "vocab.json", test_strings)

