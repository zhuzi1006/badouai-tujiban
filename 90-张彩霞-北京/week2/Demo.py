import json
import random

import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TryModel(torch.nn.Module):

    # 定义pytorch的网络模型 分两步：
    # 构造函数 1.定义网络层
    def __init__(self, input_dim, sentence_length, vocab):
        super(TryModel, self).__init__()

        # print('input_dim:', input_dim, '+len(vocab):', len(vocab))
        # embedding 层
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        # print('self.embedding:', self.embedding)
        # linear 线性层
        self.layer = nn.Linear(input_dim, input_dim)
        # 池化
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss  #loss采用均方差损失

    # 2.定义前向计算方式
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # print('x1:', x)
        x = self.embedding(x)  # input shape:(batch_size, sen_len)
        x = self.layer(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)  # input shape:(batch_size, input_dim)
        y_pred = self.activation(x)  # input shape:(batch_size, 1)
        # print('y:', y)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    # 字符集
    # chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    chars = "abcdefghijklmnopqrstuvwxyz龙师火帝鸟官人皇始制文字乃服衣裳推位让国有虞陶唐吊民伐罪周发殷汤坐朝问道垂拱平章爱育黎首臣伏戎羌遐迩一体率宾归王鸣凤在竹白驹食场化被草木赖及万方";
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现时为正样本
    # if set("周白凤") & set(x):
    if set("zcx周白凤") & set(x):
        y = 1
    # 指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):  # range(15) 从0到15[0,1,..,14]
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TryModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, test_size):
    model.eval()
    x, y = build_dataset(test_size, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), test_size - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 10  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 900  # 每轮训练总共训练的样本总数
    input_dim = 20  # 每个字的维度
    sentence_length = 12  # 样本文本长度
    test_size = 400  # 测试样本个数
    vocab = build_vocab()  # 建立字表
    model = build_model(vocab, input_dim, sentence_length)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)  # 建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构建一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, numpy.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, test_size)  # 测试本轮模型结果
        log.append([acc, numpy.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log])  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log])  # 画loss曲线
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


if __name__ == "__main__":
    main()