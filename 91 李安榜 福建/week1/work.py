import random

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    '''
    建一个简单的NN。
    词嵌入 - 线性层 - dropout - 激活层 - 池化层 - 分类 - sigmoid 概率
    '''

    def __init__(self, vocab, sentence_length, input_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) +1 , input_dim)  # embedding 初始化实例  # todo 为何还要加一
        self.layer = nn.Linear(input_dim, input_dim)
        self.avepool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss  # nn.functional .mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.layer(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.avepool(x.transpose(1, 2)).squeeze()      #  todo 这个是为啥要这样?
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # todo  这边是如何计算的。
        else:
            return y_pred


def build_model(vocab, sentence_length, input_dim):
    model = TorchModel(vocab, sentence_length, input_dim)
    return model


def build_vocab():
    '''
    abc -> 123
    :param sentence_length:
    :return:
    '''
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    dict_tmp = {}
    for index, char in enumerate(chars):
        dict_tmp[char] = index + 1
    dict_tmp['unk'] = len(dict_tmp) + 1
    return dict_tmp


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set('abc') & set(x):
        y = 1
    else:
        y = 0
    x = [vocab.get(index, vocab['unk']) for index in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for s in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_y.append([y])  # todo 为什么这里需要    [y] 与后面如何进行计算
        dataset_x.append(x)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


def main():
    train_sample = 1000
    sentenct_length = 6
    batch_size = 20
    epoch_num = 20
    char_dim = 20
    lr = 0.001
    vocab = build_vocab()
    model = build_model(vocab, sentenct_length, char_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    model.train()
    for epoch in range(epoch_num):
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentenct_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print('第 %s 轮  loss:%s' % (epoch, np.mean(watch_loss)))
        acc = evalute(model, vocab, sentenct_length)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log])  # acc
    plt.plot(range(len(log)), [l[1] for l in log])  # loss
    plt.show()
    torch.save(model.state_dict(), 'model_liab.pth')
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)


def evalute(model, vocab, sentence_length):
    model.eval()
    correct = 0
    wrong = 0
    x, y = build_dataset(200, vocab, sentence_length)
    with torch.no_grad():
        y_pred = model(x)

        for y_p, y_t in zip(y_pred, y):
            if (y_p > 0.5) & (y_t == 1):
                correct += 1
            elif (y_p <= 0.5) & (y_t == 0):
                correct += 1
            else:
                wrong += 1
    print('acc : ', correct / (correct + wrong))
    return correct / (correct + wrong)


def predict(model_path, vocab_path, input_string):
    char_dim = 20
    sentence_length = 6
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    model = build_model(vocab, sentence_length, char_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = []
    for string in input_string:
        x_1 = [vocab.get(s, vocab['unk']) for s in string]
        x.append(x_1)
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))

    for i, string in enumerate(input_string):
        print(round(float(result[i])), input_string, result[i])


if __name__ == '__main__':
    main()

    # model_path = 'model_liab.pth'
    # vocab_path = 'vocab.json'
    #
    # predict(model_path, vocab_path, ['fevxdf', 'actref'])
