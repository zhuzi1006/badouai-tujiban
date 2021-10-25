# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from loader import get_data_set, get_vocab

from customize_feature_engineering import gen_raw_id_feature_map, build_dataset

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid  # sigmoid做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss  # loss采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # input shape:(batch_size, sen_len)
        x = self.layer(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x)  # input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)  # input shape:(batch_size, input_dim)
        y_pred = self.activation(x)  # input shape:(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def load_id(raw_id, sentence_length):
    dataset_x = []
    dataset_y = []
    with open(raw_id) as rf:
        for line in rf:
            item = line.strip().split(' ')
            if item < sentence_length + 1:
                continue
            dataset_x.append(item[1:sentence_length + 1])
            dataset_y.append(item[0])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    # if torch.cuda.is_available():
    #     print("gpu可以使用，迁移模型至gpu")
    #     model = model.cuda()
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(epoch_num, model, vocab, sentence_length):
    model.eval()
    test_sample = 200
    build_dataset('test_data.txt', 'test_id.txt', vocab, test_sample, sentence_length)  # 建立200个用于测试的样本
    test_data = get_data_set('test_id.txt', test_sample, shuffle=False)
    pos_len = 0
    for x in test_data:
        pos_len += torch.sum(x[1]).item()
    print("epoch_num:%s, 本次预测集中共有%d个正样本，%d个负样本" % (epoch_num, pos_len, test_sample - pos_len))
    correct, wrong = 0, 0
    for index, t_data in enumerate(test_data):
        with torch.no_grad():
            input_ids, labels = t_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                y_pred = model(input_ids)  # 不输入labels，使用模型当前参数进行预测
            for y_p, y_t in zip(y_pred, labels):  # 与真实标签进行对比
                if float(y_p) < 0.5 and int(y_t) == 0:
                    correct += 1  # 负样本判断正确
                elif float(y_p) >= 0.5 and int(y_t) == 1:
                    correct += 1  # 正样本判断正确
                else:
                    wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    gen_raw_id_feature_map('raw.txt', 'raw_id.txt', 'feature_map.txt', train_sample, sentence_length)

    vocab = get_vocab('feature_map.txt')
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)  # 建立优化器

    train_data = get_data_set('raw_id.txt', train_sample, shuffle=True)

    log = []
    for epoch in range(epoch_num):
        print("-------epoch %s-------" % epoch)
        epoch += 1
        print(epoch)
        model.train()
        watch_loss = []
        for batch, batch_data in enumerate(train_data):
            print("-------batch %s-------" % batch)
            optim.zero_grad()  # 梯度归零
            # if torch.cuda.is_available():
            #     batch_data = [d.cuda() for d in batch_data]
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            x, y = batch_data
            # 将这些数据转换成Variable类型
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, float(np.mean(watch_loss))))
        acc = evaluate(epoch_num, model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log])  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log])  # 画loss曲线
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


# # 最终预测
# def predict(model_path, vocab_path, input_strings):
#     char_dim = 20  # 每个字的维度
#     sentence_length = 6  # 样本文本长度
#     vocab = json.load(open(vocab_path, "r", encoding="utf8"))
#     model = build_model(vocab, char_dim, sentence_length)  # 建立模型
#     model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
#     x = []
#     for input_string in input_strings:
#         x.append([vocab[char] for char in input_string])  # 将输入序列化
#     model.eval()  # 测试模式，不使用dropout
#     with torch.no_grad():  # 不计算梯度
#         result = model.forward(torch.LongTensor(x))  # 模型预测
#     for i, input_string in enumerate(input_strings):
#         print(round(float(result[i])), input_string, result[i])  # 打印结果


if __name__ == "__main__":
    main()
    # test_strings = ["abvxee", "casdfg", "rqweqg", "nlkdww", "assdaa"]
    # predict("model.pth", "vocab.json", test_strings)
